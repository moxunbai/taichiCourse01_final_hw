import os
import sys
from time import time
from engine.sim_entry import SimEntry



if __name__ == '__main__':
   conf_fn=None
   if len(sys.argv)==1:
      raise Exception('lost config file! ')
   else:
      conf_fn = sys.argv[1]
   if not os.path.exists("out"):
       os.makedirs("out")


   entry=SimEntry(conf_fn)

   entry.run("out",None)
