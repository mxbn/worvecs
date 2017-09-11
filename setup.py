from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='worvecs',
      version='0.0.1',
      description='Frequency based word vectors',
      keywords = ['word vectors', 'word2vec', 'word embedding']
      url='http://github.com/mxbn/worvecs',
      author='Max Bern',
      license='MIT',
      packages=['worvecs'],
      setup_requires=['numpy'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
