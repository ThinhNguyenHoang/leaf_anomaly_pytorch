#!/bin/bash
IMAGE_TAG=us-central1-docker.pkg.dev/thesis-372616/thesis-registry/anomaly_leaf_detect:torch_base_no_morph
package_up_training_image()
{
    docker build -t $IMAGE_TAG ./ &
}

push_to_gcr()
{
    docker push $IMAGE_TAG
}

package_up_training_image
wait
push_to_gcr