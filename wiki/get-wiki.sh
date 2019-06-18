#!/bin/bash

rm -Rf ./data
mkdir data
wget --directory-prefix=data https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz
