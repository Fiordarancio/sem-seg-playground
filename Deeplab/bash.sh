#!/bin/bash
echo "Bash version: ${BASH_VERSION}"
MAX=3
echo "Using for i in {..}"
# using this won't work! Apply string substitution!
for i in {1..$MAX..1} 
  do  
    echo "var_${i}_split"
    VAR="var_${i}"
    echo $VAR
  done
# this works
echo "Using C-like for"
for (( i=1; i<=${MAX}; i++))
  do
    echo "var_${i}_split"
  done
