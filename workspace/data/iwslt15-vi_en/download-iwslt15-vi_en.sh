# This file downloads the IWSLT15 English-Vietnamese dataset. 

DATASET_ROOT=$(cd $(dirname $0) && pwd)

cd ${DATASET_ROOT}; wget http://www.cs.toronto.edu/~bojian/Downloads/NMT/iwslt15-vi_en.tar.gz; \
	tar xvzf iwslt15-vi_en.tar.gz; rm -f iwslt15-vi_en.tar.gz