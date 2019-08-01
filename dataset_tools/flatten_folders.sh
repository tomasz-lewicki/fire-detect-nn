a=1
for i in ~/AerialImages/*.jpg
do
  #new=$(printf "%04d.jpg" "$a") #04 pad to length of 4
  #mv -i -- "$i" "$new"
  #let a=a+1
  echo $i
done
