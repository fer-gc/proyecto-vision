from os import listdir, rename

directory = "P"
files = [files for files in listdir( directory ) ]

for i in range( len( files ) ):
    ext = files[i].split( "." )[-1]
    rename(f"./{directory}/{files[i]}", f"./{directory}/image_{i}.{ext}")