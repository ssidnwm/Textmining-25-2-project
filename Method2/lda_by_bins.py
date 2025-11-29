import argparse
import os
import sys
import re
import json
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from unicodedata import normalize as u_normalize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def parse_bins(bins_str: str) -> List[Tuple[int, int]]:
    bins = []
    for part in bins_str.split(','):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d{4})\s*[-~]\s*(\d{4})$", part)
        if not m:
            raise ValueError(f"Invalid bin spec: {part}. Use e.g. 2016-2018")
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        bins.append((a, b))
    if not bins:
        raise ValueError("No valid bins provided")
    return bins


def parse_k_list(k_str: Optional[str], n_bins: int, default_k: int) -> List[int]:
    if not k_str:
        return [default_k] * n_bins
    parts = [p.strip() for p in k_str.split(',') if p.strip()]
    ks = [int(p) for p in parts]
    if len(ks) == 1:
        ks = ks * n_bins
    if len(ks) != n_bins:
        raise ValueError(f"--lda_k expects {n_bins} values (or 1 broadcast), got {len(ks)}")
    return ks


def simple_normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = u_normalize('NFKC', text)
    # Map smart quotes/dashes to ASCII
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u201c', '"').replace('\u201d', '"')
    s = s.replace('\u2013', '-').replace('\u2014', '-')
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def derive_year_column(df: pd.DataFrame, year_col: Optional[str], date_col: Optional[str]) -> Tuple[pd.Series, str]:
    # If explicit year_col exists and numeric, use it
    if year_col and year_col in df.columns:
        y = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
        return y, year_col
    # If date_col exists, parse datetime then year
    if date_col and date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors='coerce')
        y = dt.dt.year.astype('Int64')
        return y, f"year_from_{date_col}"
    # Try common fallbacks
    for c in ['year', 'Year', 'YEAR', '연도', '년도', 'date', 'Date', '날짜']:
        if c in df.columns:
            if 'year' in c.lower():
                y = pd.to_numeric(df[c], errors='coerce').astype('Int64')
                return y, c
            else:
                dt = pd.to_datetime(df[c], errors='coerce')
                y = dt.dt.year.astype('Int64')
                return y, f"year_from_{c}"
    raise ValueError("Could not infer year. Provide --year_col or --date_col explicitly.")


def extract_id_columns(df: pd.DataFrame, id_col: Optional[str], title_col_guess: Optional[str] = None) -> Tuple[pd.Series, Optional[pd.Series]]:
    if id_col and id_col in df.columns:
        return df[id_col].astype(str), None
    # Try to use title as secondary info
    title_col = None
    if title_col_guess and title_col_guess in df.columns:
        title_col = title_col_guess
    else:
        for c in ['title', 'Title', '제목']:
            if c in df.columns:
                title_col = c
                break
    titles = df[title_col].astype(str) if title_col else None
    return df.index.astype(str), titles


def top_terms_from_topic(components_row: np.ndarray, vocab: List[str], topn: int) -> List[Tuple[str, float]]:
    idxs = np.argsort(components_row)[::-1][:topn]
    return [(vocab[i], float(components_row[i])) for i in idxs]


def run_lda_for_bin(df_bin: pd.DataFrame,
                    text_col: str,
                    k: int,
                    min_df: int,
                    max_df: float,
                    max_features: Optional[int],
                    random_state: int,
                    n_top_terms: int,
                    n_iter: int,
                    learning_method: str = 'batch'):
    texts = [simple_normalize(t) for t in df_bin[text_col].fillna("")]
    if len([t for t in texts if t]) < 3:
        return None

    token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z\-']{1,}\b"
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words='english',
        token_pattern=token_pattern,
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return None

    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=random_state,
        learning_method=learning_method,
        max_iter=n_iter,
        n_jobs=-1,
    )
    W = lda.fit_transform(X)  # doc-topic
    H = lda.components_        # topic-term
    vocab = vectorizer.get_feature_names_out().tolist()

    # Topics summary
    topics_rows = []
    for t_idx in range(k):
        top_terms = top_terms_from_topic(H[t_idx], vocab, n_top_terms)
        topics_rows.append({
            'topic': t_idx,
            'top_terms': "; ".join([w for w, _ in top_terms]),
        })
    topics_df = pd.DataFrame(topics_rows)

    # Doc-topic distribution
    doc_df = pd.DataFrame(W, columns=[f'topic_{i}' for i in range(k)])
    doc_df['dominant_topic'] = doc_df.idxmax(axis=1).str.replace('topic_', '', regex=False).astype(int)

    return {
        'vectorizer': vectorizer,
        'lda': lda,
        'doc_topic': doc_df,
        'topics': topics_df,
        'vocab': vocab,
    }


def main():
    ap = argparse.ArgumentParser(description='Run LDA over time bins (e.g., yearly or multi-year) to analyze trends.')
    ap.add_argument('--excel', required=True, help='Path to Excel file with abstracts')
    ap.add_argument('--sheet', default=None, help='Sheet name (optional)')
    ap.add_argument('--text_col', default='abstract', help='Column with abstract text')
    ap.add_argument('--year_col', default=None, help='Column with year (int)')
    ap.add_argument('--date_col', default=None, help='Column with parseable dates (if year missing)')
    ap.add_argument('--id_col', default=None, help='Optional unique ID column')
    # Manual bins (kept for backward compatibility). Ignored if --bin_size is provided.
    ap.add_argument('--bins', default='2016-2018,2019-2021,2022-2024', help='Comma-separated year bins e.g. 2016-2018,2019-2021 (ignored if --bin_size is set)')
    # Auto binning options
    ap.add_argument('--bin_size', type=int, default=None, help='If set, auto-generate contiguous bins of this size from min..max year (use 1 for yearly bins)')
    ap.add_argument('--year_min', type=int, default=None, help='Optional lower bound year for auto bins (defaults to min year in data)')
    ap.add_argument('--year_max', type=int, default=None, help='Optional upper bound year for auto bins (defaults to max year in data)')
    ap.add_argument('--lda_k', default=None, help='Topics per bin (comma-separated) or single value')
    ap.add_argument('--default_k', type=int, default=6, help='Default topics if --lda_k not given')
    ap.add_argument('--min_df', type=int, default=2)
    ap.add_argument('--max_df', type=float, default=0.9)
    ap.add_argument('--max_features', type=int, default=None)
    ap.add_argument('--n_top_terms', type=int, default=15)
    ap.add_argument('--n_iter', type=int, default=20)
    ap.add_argument('--random_state', type=int, default=42)
    ap.add_argument('--outdir', default=os.path.join('outputs', 'trends', 'lda_by_bin'))
    args = ap.parse_args()

    # Load data
    if not os.path.exists(args.excel):
        print(f"ERROR: Excel not found: {args.excel}", file=sys.stderr)
        sys.exit(2)
    try:
        raw = pd.read_excel(args.excel, sheet_name=(args.sheet if args.sheet is not None else 0))
        df = raw if isinstance(raw, pd.DataFrame) else next(iter(raw.values()))
    except Exception as e:
        print(f"ERROR reading Excel: {e}", file=sys.stderr)
        sys.exit(2)

    if args.text_col not in df.columns:
        # Try common alternatives
        candidates = [c for c in ['abstract', 'Abstract', '초록', '내용'] if c in df.columns]
        if not candidates:
            print(f"ERROR: text column '{args.text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)
        args.text_col = candidates[0]

    year_series, year_label = derive_year_column(df, args.year_col, args.date_col)
    df['_year'] = year_series
    df = df[~df['_year'].isna()].copy()
    df['_year'] = df['_year'].astype(int)
    if df.empty:
        print("ERROR: no rows with valid year.", file=sys.stderr)
        sys.exit(2)

    ids, titles = extract_id_columns(df, args.id_col)
    df['_doc_id'] = ids
    if titles is not None:
        df['_title'] = titles

    # Determine bins: prefer auto bins when --bin_size is given
    if args.bin_size is not None and args.bin_size > 0:
        yr_min = int(df['_year'].min()) if args.year_min is None else int(args.year_min)
        yr_max = int(df['_year'].max()) if args.year_max is None else int(args.year_max)
        if yr_min > yr_max:
            yr_min, yr_max = yr_max, yr_min
        bins = []
        start = yr_min
        while start <= yr_max:
            end = min(start + args.bin_size - 1, yr_max)
            bins.append((start, end))
            start = end + 1
        bin_mode = f'auto_size_{args.bin_size}'
    else:
        bins = parse_bins(args.bins)
        bin_mode = 'manual'
        yr_min, yr_max = int(df['_year'].min()), int(df['_year'].max())

    ks = parse_k_list(args.lda_k, len(bins), args.default_k)

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_out = os.path.join(args.outdir, f"run_{run_ts}")
    ensure_dir(base_out)

    manifest = {
        'excel': os.path.abspath(args.excel),
        'sheet': args.sheet,
        'text_col': args.text_col,
        'year_source': year_label,
        'bins': bins,
        'bin_mode': bin_mode,
        'bin_size': args.bin_size,
        'year_min': yr_min,
        'year_max': yr_max,
        'k_list': ks,
        'min_df': args.min_df,
        'max_df': args.max_df,
        'max_features': args.max_features,
        'n_top_terms': args.n_top_terms,
        'n_iter': args.n_iter,
        'random_state': args.random_state,
        'created': run_ts,
    }
    with open(os.path.join(base_out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary_rows = []
    for (bin_idx, (a, b)) in enumerate(bins):
        k = ks[bin_idx]
        sel = (df['_year'] >= a) & (df['_year'] <= b)
        df_bin = df.loc[sel].reset_index(drop=True)
        bin_dir = os.path.join(base_out, f"bin_{a}_{b}")
        ensure_dir(bin_dir)

        if df_bin.empty:
            with open(os.path.join(bin_dir, 'SKIPPED.txt'), 'w', encoding='utf-8') as f:
                f.write(f"No documents in bin {a}-{b}. Skipped.\n")
            summary_rows.append({'bin': f'{a}-{b}', 'docs': 0, 'topics': 0, 'status': 'skipped'})
            continue

        result = run_lda_for_bin(
            df_bin=df_bin,
            text_col=args.text_col,
            k=k,
            min_df=args.min_df,
            max_df=args.max_df,
            max_features=args.max_features,
            random_state=args.random_state,
            n_top_terms=args.n_top_terms,
            n_iter=args.n_iter,
        )
        if result is None:
            with open(os.path.join(bin_dir, 'SKIPPED.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Insufficient data/tokens for LDA in bin {a}-{b}.\n")
            summary_rows.append({'bin': f'{a}-{b}', 'docs': int(df_bin.shape[0]), 'topics': 0, 'status': 'skipped'})
            continue

        topics_df = result['topics']
        doc_topic_df = result['doc_topic']

        # Attach IDs/titles/years to doc_topic
        enriched = pd.concat([
            df_bin[['_doc_id'] + (['_title'] if '_title' in df_bin.columns else []) + ['_year']].reset_index(drop=True),
            doc_topic_df.reset_index(drop=True)
        ], axis=1)

        topics_df.to_csv(os.path.join(bin_dir, 'topics.csv'), index=False, encoding='utf-8-sig')
        enriched.to_csv(os.path.join(bin_dir, 'doc_topic.csv'), index=False, encoding='utf-8-sig')

        # Also save vocabulary and topic-term weights for transparency
        vocab = result['vocab']
        pd.DataFrame({'term': vocab}).to_csv(os.path.join(bin_dir, 'vocab.csv'), index=False, encoding='utf-8-sig')
        # components_ can be large; save as compressed npz plus a small preview csv
        np.savez_compressed(os.path.join(bin_dir, 'topic_term.npz'), components=result['lda'].components_)
        topics_df.to_csv(os.path.join(bin_dir, 'topics_top_terms.csv'), index=False, encoding='utf-8-sig')

        summary_rows.append({'bin': f'{a}-{b}', 'docs': int(df_bin.shape[0]), 'topics': int(k), 'status': 'ok'})

    pd.DataFrame(summary_rows).to_csv(os.path.join(base_out, 'summary.csv'), index=False, encoding='utf-8-sig')
    print(f"LDA by bins complete. Outputs at: {base_out}")


if __name__ == '__main__':
    main()
