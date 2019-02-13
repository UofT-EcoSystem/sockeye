# This file downloads the IWSLT15 English-Vietnamese dataset. 

DATASET=iwslt15-vi_en
DATASET_ROOT=$(cd $(dirname $0) && pwd)

cd ${DATASET_ROOT}; wget http://www.cs.toronto.edu/~bojian/Downloads/NMT/${DATASET}.tar.gz; \
	tar xvzf ${DATASET}.tar.gz; rm -f ${DATASET}.tar.gz