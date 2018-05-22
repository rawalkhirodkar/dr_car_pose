mkdir temp
cp target/* temp/
cp view1/* temp/
cp view2/* temp/
cp view3/* temp/

mkdir stitched

cd temp
for img in ../target/*; do           # loop over the directories
    (                        # run in a subshell ...
        name=$(basename "$img")
        img_id=${name::-8}
        echo "$img_id"
        convert \( "$img_id"_det.jpg "$img_id"_v1.jpg +append  \) \( "$img_id"_v2.jpg "$img_id"_v3.jpg +append \) -append ../stitched/"$img_id".jpg
    )
done

rm -f *
cd ..
rm -d temp

cd stitched

ls | cat -n | while read n f; do mv "$f" `printf "%04d.jpg" $n`; done 



ffmpeg -r 1 -i %04d.jpg -r 25 -vcodec mpeg4 -pix_fmt yuv420p -qp 0 ../scene2.mp4