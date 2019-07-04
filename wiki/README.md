## Example scripts

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
