import random
import pickle

random.seed(1550)

user_num = 1000
item_num = 50
tag_num = 5
each_tag_num = 10

user_embed_list = []
item_embed_list = []
tag_item_list = []

for tag_id in range(tag_num):
	cur_item = [_ + tag_id * each_tag_num for _ in range(each_tag_num)]
	tag_item_list.append(cur_item)
# print(tag_item_list)
for item_id in range(item_num):
	item_embed = [0 for _ in range(item_num)] 
	item_embed[item_id] = 1
	item_embed_list.append(item_embed)
# print(item_embed_list)

for user_id in range(user_num):
	user_embed = [0. for _ in range(item_num)] 
	for tag_id in range(tag_num):
		tag_score = random.random()
		for item_id in tag_item_list[tag_id]:
			user_embed[item_id] = tag_score
	user_embed_list.append(user_embed)
	# print(f"pos_tag {pos_tag}, neg_tag {neg_tag} \n {user_embed}")
# print(user_embed_list)

synthetic_data = (tag_item_list, item_embed_list, user_embed_list)
with open("synthetic2.pkl", "wb") as f:
	pickle.dump(synthetic_data, f)