from indexer import RawTextIndexer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Indexer for raw text files')
    parser.add_argument("--d", default='test docs', type=str, help="Name of directory with TXT files")
    parser.add_argument("--n", default='3', type=int, help="Number of most frequent words")
    args = parser.parse_args()
    txtDir = args.d
    top_n = args.n

    ixer = RawTextIndexer(txtDir)
    ixer.index_files()
    output = ixer.format_top_n(top_n)

    for line in output:
        print(' | '.join(line))
        print('-' * 80, "\n")


if __name__ == "__main__":
    main()