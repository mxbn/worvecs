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
         built in 19 min

 driving -> car, sailing -> ?
         sailing
         car
         motorcycle
         yacht
         catamaran
         trainer
         racing
         tourer
         touring
         boat

 taxi -> ?
         rail
         freight
         buses
         commuter
         passenger
         line
         train
         trains
         transit
```

<sup>1</sup> As ran on a 32GB RAM 16 core (32 thread) computer.  
