i=0
for file in ./FireVideosCurated/*
do
	fname_w_ext=$(basename $file)
	name="${fname_w_ext%.*}"
	#mkdir ./FireImages/$name
	duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file)
	ffmpeg -i $file -vf fps=0.25 ./FireImagesCurated/$name-%d.jpg
done

