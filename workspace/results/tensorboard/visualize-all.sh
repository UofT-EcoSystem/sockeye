#!/bin/bash

./visualizer.py --csv-prefix=iwslt15-vi_en-groundhog-500 --metric="speed" --metric-unit="Samples/s"
./visualizer.py --csv-prefix=iwslt15-vi_en-groundhog-500 --metric="perplexity" 
./visualizer.py --csv-prefix=iwslt15-vi_en-groundhog-500 --metric="memory_usage" --metric-unit="MB" 