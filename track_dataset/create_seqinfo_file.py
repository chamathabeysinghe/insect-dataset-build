import os

dirs = ['sample2', 'sample3']
DATASET_BASE_DIR = '/Users/cabe0006/Projects/monash/dataset-build/sample_data'
DATASET_DIR = os.path.join(DATASET_BASE_DIR, 'track_dataset')

for d in dirs:
    sample_dir = os.path.join(DATASET_DIR, d)
    file_dir = os.path.join(sample_dir, 'seqinfo.ini')
    content = "[Sequence]\nname={}\nimDir=img1\nframeRate=6\nseqLength=10\nimWidth=4096\nimHeight=2168\nimExt=.jpg".format(
        d)

    f = open(file_dir, "w")
    f.write(content)
    f.close()



