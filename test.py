import re


def write(a,b):
    with open(a, 'w') as f:
        f.seek(0)  
        f.truncate()
        for i in b:
            f.write(i+'\n')
    f.close()
    
    
path='data/DBP15K/D_W_100K_V1/'
ent_path=path+'train_links'
ent_path1=path+'ent_ids_1'
ent_path2=path+'ent_ids_2'
name_path1=path+'ent_names_1'
name_path2=path+'ent_names_2'
triple_path1=path+'triples_1'
triple_path2=path+'triples_2'
entity_link_path=path+'ref_ent_ids'
i=0
j=70000
entity_list1=[]
entity_list2=[]
entity_name_list1=[]
entity_name_list2=[]
dict1={}
dict2={}
with open(ent_path, 'r') as f:
    for line in f:
        info = line.strip().split('\t')
        entity1=str(i)+'\t'+info[0]
        entity2=str(j)+'\t'+info[1]
        entity_name1=str(i)+'\t'+info[0].split('/')[-1].replace('_',' ').replace('-',' ')
        entity_name2=str(j)+'\t'+info[0].split('/')[-1].replace('_',' ').replace('-',' ')
        # print(deepltrans(info[1].split('/')[-1]))
        entity_list1.append(entity1)
        entity_list2.append(entity2)
        entity_name_list1.append(entity_name1)
        entity_name_list2.append(entity_name2)
        dict1[info[0]]=i
        dict2[info[1]]=j
        i=i+1
        j=j+1
f.close()
write(ent_path1,entity_list1)
write(ent_path2,entity_list2)
write(name_path1,entity_name_list1)
write(name_path2,entity_name_list2)
entity_link=[]
with open(ent_path, 'r') as f:
    for line in f:
        info = line.strip().split('\t')
        link=str(dict1[info[0]])+'\t'+str(dict2[info[1]])
        entity_link.append(link)
f.close()
write(entity_link_path,entity_link)
triple1_path=path+'rel_triples_1'
dict3={}
i=0
with open(triple1_path, 'r') as f:
    for line in f:
        info = line.strip().split('\t')
        if info[1] not in dict3:
            dict3[info[1]]=i
            i=i+1
f.close()
triple_list1=[]
with open(triple1_path, 'r') as f:
    for line in f:
        info = line.strip().split('\t')
        if info[0] in dict1 and info[2] in dict1:
            triple1=str(dict1[info[0]])+'\t'+str(dict3[info[1]])+'\t'+str(dict1[info[2]])
            triple_list1.append(triple1)
f.close()
write(triple_path1,triple_list1)

triple2_path=path+'rel_triples_2'
dict4={}
i=0
with open(triple2_path, 'r') as f:
    for line in f:
        info = line.strip().split('\t')
        if info[1] not in dict4:
            dict4[info[1]]=i
            i=i+1
f.close()
triple_list2=[]
with open(triple2_path, 'r') as f:
    for line in f:
        info = line.strip().split('\t')
        if info[0] in dict2 and info[2] in dict2:
            triple2=str(dict2[info[0]])+'\t'+str(dict4[info[1]])+'\t'+str(dict2[info[2]])
            triple_list2.append(triple2)
f.close()
write(triple_path2,triple_list2)
# new_name_list=[]
# with open(name_path2, 'r') as f:
#     for line in f:
#         info = line.strip()
#         new_name=info[0:5]+'\t'+info[5:].replace('\t','').replace('_',' ').replace('-',' ')
#         new_name_list.append(new_name)
# f.close()
# write(path+'ent_names_3',new_name_list)