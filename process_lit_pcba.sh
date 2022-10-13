for target in *; do
cd $target
pymol -c process.pml
cd ..
done
