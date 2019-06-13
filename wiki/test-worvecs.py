import sys
sys.path.append("..")
from worvecs import worvecs

print('\n\nencoding 0:')
model = worvecs()
model.load('./data/wiki-worvecs-0.txt.gz')

print('\ndriving -> car, sailing -> ?:')
for s in model.similarRelations('driving', 'car', 'sailing')[0]:
    print('\t%s' % s)

print('\ntaxi:')
for s in model.similarWords('bus')[0]:
    print('\t%s' % s)

print('\n\n\nencoding 1:')
model = worvecs()
model.load('./data/wiki-worvecs-1.txt.gz')

print('\ndriving -> car, sailing -> ?:')
for s in model.similarRelations('driving', 'car', 'sailing')[0]:
    print('\t%s' % s)

print('\ntaxi:')
for s in model.similarWords('bus')[0]:
    print('\t%s' % s)
print('\n')
