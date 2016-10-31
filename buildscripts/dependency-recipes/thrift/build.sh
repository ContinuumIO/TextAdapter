#!/bin/sh

./configure --prefix=$PREFIX --with-boost=$PREFIX --disable-tests --with-cpp \
    --without-qt4 \
    --without-qt5 \
    --without-c_glib \
    --without-csharp \
    --without-java \
    --without-erlang \
    --without-nodejs \
    --without-lua \
    --without-python \
    --without-perl \
    --without-php \
    --without-php_extension \
    --without-ruby \
    --without-haskell \
    --without-go \
    --without-haxe \
    --without-d
make
make install
