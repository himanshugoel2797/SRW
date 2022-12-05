.PHONY: python clean

MODE ?= 0 

python: srwlpy.so

srwlpy.so:
	MODE=$(MODE) python setup.py build_ext --build-lib='../gcc'
	cp ../gcc/srwlpy*.so ../../clients/python/srwpy/srwlpy.so
	rm -rf build

clean:
	rm -rf build
	rm -f ../gcc/srwlpy*.so
	rm -f ../../clients/python/srwpy/srwlpy*.so