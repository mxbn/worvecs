## Sample use case

Run:  
1. `./get-wiki.sh` to download latest wiki abstracts.  
2. `python preprocess.py` to extract and tokenize the abstracts.  
3. `python test-worvecs.py` to build and test word vectors.  

Expected output from step 3<sup>1</sup>:  
 ```
loading text...
        loaded 4848075 sentences
building a model...
        built in 59 min

driving -> car, sailing -> ?
        yacht
        sailing
        fastnet
        yachting
        regattas
        ship
        boat
        cruise
        shipyard
        keelboat

taxi -> ?
        rail
        transit
        buses
        train
        railways
        commuter
        routes
        tram
        trains
```

<sup>1</sup> As ran on a 16GB RAM +16GB swap, 4 core (4 thread) CPU.  

### Compared to fasttext  

1. `zcat data/abstract.txt.gz > data/abstract.txt`  
2. `git clone https://github.com/facebookresearch/fastText`  
3. `cd fastText/ && make && cd ../`  
4. `./fastText/fasttext skipgram -input data/abstract.txt -output data/fasttext -maxn 0`  
5. `./fastText/fasttext analogies data/fasttext.bin`  
  5.1. `sailing driving car`  
```
yacht 0.725373
yachts 0.694678
catamarans 0.652838
sloop-rigged 0.638242
yachting 0.626513
staysail 0.623325
dinghy 0.621671
rowing 0.616934
boat 0.613638
single-masted 0.612197
```
6. `./fastText/fasttext nn data/fasttext.bin `  
  6.1. `taxi`  
```
taxicabs 0.709916
taxicab 0.701613
minibuses 0.672338
long-haul 0.663033
vehicle 0.661264
for-hire 0.655056
beeline 0.654826
limousines 0.648476
no-frills 0.648117
pronto 0.643631
```  
