from dataset.mscoco import COCOJeopardy 
import random

a = COCOJeopardy()
Length = len(a)

print(a[4])
COCO_TEST_IMAGES = "./COCO_TEST_IMAGES/"
random_ints = [random.randint(0, Length) for _ in range(10)]
for i in random_ints:
    im, q = a[i]
    print(q)
    im.save(COCO_TEST_IMAGES + q + ".png")
