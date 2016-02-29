#!/bin/bash

ARCHIVE_URL="http://hci.iwr.uni-heidelberg.de/Download/lightfield_archive"

# FIRST PART: DOWNLOAD RAW LIGHTFIELD DATA AND PRECOMPUTED EDGE WEIGHTS
# The first part for Each light field requires at least 250 MB storage.
echo
echo "*********************************"
echo "***          PART 1           ***"
echo "*********************************"
echo

# download script for first part
function download_lf_part1 {
    echo
    echo "*********************************"
    echo "***     "$1 " : "$2
    echo "*********************************"
    echo
    mkdir -p $1/$2
    cd $1/$2
    if [ ! -s lf.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/lf.h5
    fi
    if [ ! -s edge_weights.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/edge_weights.h5
    fi
    cd ../../
    echo
    echo
}

# download light fields (you can of course uncomment those not needed)
# 1A: ray-traced light fields
download_lf_part1 blender buddha
# download_lf_part1 blender buddha2
# download_lf_part1 blender horses
# download_lf_part1 blender medieval
# download_lf_part1 blender monasRoom
# download_lf_part1 blender papillon
# download_lf_part1 blender stillLife

# # 1B: gantry light fields
# download_lf_part1 gantry maria
# download_lf_part1 gantry couple
# download_lf_part1 gantry cube
# download_lf_part1 gantry pyramide
# download_lf_part1 gantry statue
# download_lf_part1 gantry transparency


# SECOND PART: DOWNLOAD DATA FOR LIGHT FIELD SEGMENTATION
# Careful: The second part for each light field requires a few Gigabyte.
echo
echo "*********************************"
echo "***          PART 2           ***"
echo "*********************************"
echo

# download script for first part
function download_lf_part2 {
    echo
    echo "*********************************"
    echo "***     "$1 " : "$2
    echo "*********************************"
    echo
    mkdir -p $1/$2
    cd $1/$2


    # ... disparity_local.h5
    if [ ! -s disparity_local.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/disparity_local.h5
    fi

    # ... labels.h5
    if [ ! -s labels.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/labels.h5
    fi

    # ... feature_depth_probabilities.h5
    if [ ! -s feature_depth_probabilities.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/feature_depth_probabilities.h5
    fi

    # ... feature_gt_depth_probabilities.h5
    if [ ! -s feature_gt_depth_probabilities.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/feature_gt_depth_probabilities.h5
    fi

    # ... feature_single_view_probabilities.h5
    if [ ! -s feature_single_view_probabilities.h5 ]
    then
    wget $ARCHIVE_URL/$1/$2/feature_single_view_probabilities.h5
    fi
    cd ../../
    echo
    echo
}

# download light field data (you can of course uncomment those not needed)
# 2A: ray-traced light fields
# download_lf_part2 blender buddha
# download_lf_part2 blender horses
# download_lf_part2 blender papillon
# download_lf_part2 blender stillLife