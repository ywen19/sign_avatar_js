This folder contains preprocess modules to clean up the vocab folder.
Raw data is gained from: https://www.signbsl.com/sign/dictionary.


For each preprocess and cleanup modules,
we provide dry-run and actual run flags, so that users could check
what operations are going to be done before actually processing large
amount of data. 


All these cleanup happens on textual contents, rather than including
motion-related processing steps.

Steps to run cleanup process are documented as follows.
---


## Overall Checks
1. ```all_folder_names.py```:   
Write all immediate subfolder names into a JSON file;
So that for later steps, we could always look back to this JSON as reference.    
<mark><b>Always re-run this to update the vocab catalogue after a cleanup is done.</b></mark>

2. ```rm_empty.py```:  
Remove vocab folders that contain no video files. 

3. ```scan_illegal_symbols.py```:  
Scan all illegal symbols on the level of vocab folders.  
The script treats any character that is not alphanumeric and not an underscore (_) as illegal.  
In actual practice, we <mark>do dry-run first</mark> to check the logs manually. Since the illegal symbols here contain different meanings and cases, that acquired to be cleaned differently.
Through the log, we detected apostrophe, brackets, comma and some other cases.  
<mark><b>Please manual check logs if in future the data is extended.</b></mark>
---


## Illegal Symbol Cleanup
For clearing up illegal symbols, we add a handling on the cleanedup vocab folder names:
before actually renaming the folder, we first check if any vocab folders that have the same name as the rename target; If exists, we will append an enumerate index suffix to the target name.  
This is future used after textual content cleanup is done, and when merge needs to applied to these duplicant vocab folders.  
<mark><b>Please dry-run first and make sure operations documented in the output logs are exactly what needs to be applied.</b></mark>  
This is because for below cleanup, we have many data-sepecific cleanup rules.
If the raw data is changed, the script needs to be adjusted accordingly.  
So please also refer to the check log from ```scan_illegal_symbols.py``` first before you directly run this script, even before dry-run. 


1. ```remove_special_symbols_nonbrackets```:  
Remove all special symbols that does not fall into 2, 3, and 4.  
Basically we remove '.' and '-', and replace '-' with '_'.


2. ```remove_apostrophe.py```:  
Directly remove all apostrophe. For example, 'John's' will become 'Johns'. If 'Johns' already exists, then the vocab folder will be renamed as 'Johns_1' (such duplicant handling is applied to all below modules).

3. ```remove_brackets.py```:   
We manually check all cases that contain brackets.  
There are three cases:  
   * removes the parenthetical part entirely: Water_(General) -> Water
   * removes the brackets but keeps the text inside: Change_(Money) -> Change_Money
   * split_alias: field_effect_transistor_(FET) -> field_effect_transistor and FET

4. ```remove_comma.py```:   
We manually check all cases that contain comma.     
There are three cases:  
   * removes comma: Hello,_my_name_is -> Hello_my_name_is
   * rewrite: angry,_very -> very_angry
   * split_alias: Nothing,_Nobody -> Nothing and Nobody

5. ```trim_edge_underscores.py```:    
Since we apply renaming from 1-4, which may result in underscores appearing at the start or end of a vocab, we need to therefore run this post steps as post cleanup.  
Same goes for step 6.

6. ```collapse_repeated_underscores.py```
---


## Final Cleanup and Merge
1. ```lowercase_folders.py```

2. ```all_folder_names.py```:  
Essential to run here, since we need to update the scanned vocabs for later duplicants handling. 

3. ```lowercase_duplicants_json.py```:  
Run this before the duplicate comparison to guarantee that all keys and values already stored in duplicants.JSON use the same lowercase format as the folders.  
It also merges groups whose keys collide after lowercasing. For example:  
"Hello": ["Hello", "Hello_1"]  
"hello": ["hello", "hello_2"]  

4. ```find_all_duplicants.py```:   
Compares the final lowercase folder names against the final lowercase duplicate-groups JSON and finds groups such as:  
user  
user_1  
user_2  

5. ```merge_duplicate_groups.py```:  
Merge duplicate vocab folders and also rename the files in the merged folder with enumerate index suffix.  




