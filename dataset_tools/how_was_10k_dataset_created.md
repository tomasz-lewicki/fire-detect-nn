First download the entire playlist off youtube: (50 positive videos)

```shell
youtube-dl --yes-playlist https://www.youtube.com/playlist?list=PLYtFiJNWfdpASBUHq5SAufQoK0I_DMlT6 --playlist-start 1 --restrict-filenames
```

Then convert videos into images at 0.25fps (every 4 seconds)

```bash
i=0
for file in ./FireVideosCurated/*
do
	fname_w_ext=$(basename $file)
	name="${fname_w_ext%.*}"
	#mkdir ./FireImages/$name
	duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $file)
	ffmpeg -i $file -vf fps=0.25 ./FireImagesCurated/$name-%d.jpg
done
```

The result is 4233 images