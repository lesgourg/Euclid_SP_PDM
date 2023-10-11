BASEDIR=$(dirname "$0")
g++ -shared -o $BASEDIR/euclid_photo.so $BASEDIR/euclid_photo.cpp -I $BASEDIR -fPIC -O3 -lgomp -fopenmp
