#!/bin/bash

mkdir -p __data__/opensubs
cd __data__/opensubs
wget -O en.tar.gz http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/en.tar.gz
tar -xf en.tar.gz
rm en.tar.gz
