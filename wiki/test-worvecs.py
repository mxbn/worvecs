import sys
sys.path.append("..")
from worvecs import worvecs

print('\n\nmodel 0:')
model = worvecs()
model.load('./data/wiki-worvecs-0.txt.gz')

print('\ncar -> driving, motorcycle -> ?:')
for s in model.similarRelations('car', 'driving', 'motorcycle')[0]:
    print('\t%s' % s)

print('\nbus:')
for s in model.similarWords('bus')[0]:
    print('\t%s' % s)

print('\n\n\nmodel 1:')
model = worvecs()
model.load('./data/wiki-worvecs-1.txt.gz')

print('\ncar -> driver, motorcycle -> ?:')
for s in model.similarRelations('car', 'driving', 'motorcycle')[0]:
    print('\t%s' % s)

print('\nbus:')
for s in model.similarWords('bus')[0]:
    print('\t%s' % s)
print('\n')
