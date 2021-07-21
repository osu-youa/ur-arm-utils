from rosbag import Bag
import os

folder = '/home/main/data/data_collection_oct2020'
bags = [f for f in os.listdir(folder) if f.endswith('.bag') and not f.endswith('_filtered.bag')]

for bag in bags:
    new_bag = bag.replace('.bag', '_filtered.bag')
    assert new_bag != bag
    with Bag(os.path.join(folder, new_bag), 'w') as outbag:
        for info in Bag(os.path.join(folder, bag)).read_messages():
            topic = info[0]
            if topic.startswith('/camera/color') or topic.startswith('/camera/depth'):
                continue
            outbag.write(*info)