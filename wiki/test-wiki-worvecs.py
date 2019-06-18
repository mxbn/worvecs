import sys, time, gzip, os
import regex as re
sys.path.append("..")
import worvecs

model_fname = './data/wiki-worvecs.txt.gz'

if __name__ == '__main__':
    if os.path.isfile(model_fname):
        m = worvecs.model()
        m.load(model_fname)
    else:
        print('loading text...')
        sentences = []
        sentence_finder = re.compile(r'.*?[\.\?!]|.*$')
        with gzip.open('./data/abstract.txt.gz', 'rt', encoding='utf-8', \
            errors='replace') as f:
            for l in f:
                for sentence in sentence_finder.findall(l):
                    if len(sentence) > 1:
                        sentences.append(sentence.strip().split(' '))
        print('\tloaded %d sentences' % len(sentences))

        t0 = time.time()
        print('building a model...')
        m = worvecs.model(sentences)
        m.save(model_fname)
        print('\tbuilt in %d min' % ((time.time() - t0)/60))

    print('\ndriving -> car, sailing -> ?')
    sim = m.similarRelations('driving', 'car', 'sailing')
    if sim is not None:
        for s in sim[0]:
            print('\t%s' % s)

    print('\ntaxi -> ?')
    sim = m.similarWords('bus')
    if sim is not None:
        for s in sim[0]:
            print('\t%s' % (s))
