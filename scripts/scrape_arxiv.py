import arxiv
import requests
from langchain_community.document_loaders import PyPDFLoader
import os

def search_arxiv(query, max_results=100):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    # Search.results() is deprecated in arxiv 2.x; the Client owns paging/retry.
    client = arxiv.Client()

    papers = []
    for result in client.results(search):
        paper_info = {
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'summary': result.summary,
            'pdf_url': result.pdf_url,
            'published': result.published,
            'categories': result.categories
        }
        papers.append(paper_info)
    return papers

def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {pdf_url}")

def extract_text_from_pdf(pdf_path):
    print(f"Parsing {pdf_path} with PyPDFLoader...")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Failed to parse {pdf_path}: {e}")
        return ""

def write_string_to_file(filename, content):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Successfully wrote content to {filename}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download arXiv papers and extract their text.")
    parser.add_argument("--query", default="machine learning")
    parser.add_argument("--count", type=int, default=5, help="How many papers to download.")
    parser.add_argument("--start-index", type=int, default=1,
                        help="First paperN index to write, so repeat runs can extend the corpus.")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    print(f"Searching ArXiv for: {args.query!r}")
    papers = search_arxiv(args.query, max_results=args.count)

    for offset, paper in enumerate(papers[:args.count]):
        idx = args.start_index + offset
        print(f"[{offset + 1}/{args.count}] {paper['title']}")
        pdf_path = os.path.join(data_dir, f"paper{idx}.pdf")
        download_pdf(paper['pdf_url'], pdf_path)

        if os.path.exists(pdf_path):
            text = extract_text_from_pdf(pdf_path)
            if text:
                write_string_to_file(os.path.join(data_dir, f'pdf_{idx}.txt'), text)

    print("\nDone. Now run: python scripts/build_index.py --rebuild")
