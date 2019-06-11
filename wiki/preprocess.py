import gzip, time
import regex as re
from multiprocessing import Pool, cpu_count

n_threads = cpu_count()

abstract_finder = re.compile(r'<abstract>(.*?)</abstract>')
empty_abstract_finder = re.compile(r'\|?[\s_a-z0-9]+=|=\s*=\s*[a-z]+\s*=\s*=|.*?may\s*refer\s*to\s*\:')
html_tags = re.compile(r'<.*?>|&#?\w{2,6};?')
urls = re.compile(r'((?:https?\://(?:www\.|en\.|fr\.|uk\.|au\.)?|www\.)([^\s\.]+)(?:\.[^\s/]+)?\.([a-z]{2,5})(?:[^\s]*))')
token_finder = re.compile(r'\w+(?:[\'\-&\+]\w+)?|[^\w\s\]\[\(\)"\|{}]')

def getWords(text):
    m = abstract_finder.match(text)
    if m is not None:
        t = str(m.groups()[0]).lower()
        mm = empty_abstract_finder.match(t)
        if mm is None:
            t = html_tags.sub(' ', t)
            t = urls.sub(' ', t)
            tokens = token_finder.findall(t)
            if len(tokens) > 3:
                return ' '.join(tokens) + '\n'
    return ''

if __name__ == '__main__':
    print('processing...')
    t0 = time.time()
    with gzip.open('./data/enwiki-latest-abstract.xml.gz', 'rt', encoding='utf8') as f,\
        gzip.open('./data/abstract.txt.gz', 'wt', encoding='utf8') as fout,\
        Pool(n_threads) as p:
            for abstract in p.imap_unordered(getWords, f):
                fout.write(abstract)
    print('finished: %d min' % ((time.time()-t0)/60))
