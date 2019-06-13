import sys
sys.path.append("..")
from worvecs import worvecs

print('model 0:')
model = worvecs()
model.load('./data/wiki-worvecs-0.txt.gz')

print('\nking -> queen, father -> ?:')
for s in model.similarRelations('car', 'driver', 'motorcycle')[0]:
    print('\t%s' % s)

print('\nbus:')
for s in model.similarWords('bus')[0]:
    print('\t%s' % s)

print('model 1:')
model = worvecs()
model.load('./data/wiki-worvecs-1.txt.gz')

print('\nking -> queen, father -> ?:')
for s in model.similarRelations('king', 'queen', 'father')[0]:
    print('\t%s' % s)

print('\nbus:')
for s in model.similarWords('bus')[0]:
    print('\t%s' % s)
