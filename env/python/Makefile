.PHONY: python clean

MODE ?= 0 

python: srwlpy.so

srwlpy.so:
	MODE=$(MODE) python setup.py build_ext --build-lib='../src'
	cp ../src/srwlpy*.so ./srwpy/srwlpy.so
	rm -rf build

clean:
	rm -rf build
	rm -f ../src/srwlpy*.so
	rm -f ../../clients/python/srwpy/srwlpy*.so