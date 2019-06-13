## Example scripts

Run:  
 - `./get-wiki.sh` to download wiki abstracts  
 - `python preprocess.py` to extract and tokenize the abstracts  
 - `python build-wiki-worvecs.py` to buld word vectors  
 - `python test-worvecs.py` to test word vectors  

 Expected output:
 ```
 encoding 0:

 driving -> car, sailing -> ?:
  	yacht
  	airshow
  	sailing
  	oshkosh
  	regatta
  	round-the
  	racer
  	racing
  	raced
  	rowing

taxi:
  	buses
  	trams
  	streetcars
  	trolley
  	taxis
  	taxicabs
  	trolleybuses
  	streetcar
  	cabs



encoding 1:

driving -> car, sailing -> ?:
  	sailing
  	car
  	yacht
  	motorcycle
  	cruise
  	boat
  	racing
  	cargo
  	trainer
  	seaplane

taxi:
  	rail
  	buses
  	commuter
  	freight
  	train
  	line
  	transit
  	operated
  	trains
```
