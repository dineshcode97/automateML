#!/usr/bin/env python
# coding: utf-8

# In[110]:


import os
filesize = os.path.getsize("accuracy.txt")
if filesize == 0:
    print("The file is empty: " + str(filesize))
else:
    a_file = open("val.txt", "r")
    lines = a_file.readlines()
    a_file.close()
    del lines[0]
    new_file = open("val.txt", "w+")
    for line in lines:
        new_file.write(line)
    new_file.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




