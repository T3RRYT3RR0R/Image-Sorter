# update files + update manifest
## update files are headed with a changelog explaining changes by version.

The [self-update](https://github.com/T3RRYT3RR0R/Image-Sorter/blob/main/updates/update_297_image-sorter.py) function adheres to the following control flow:

 - define path constants
 - load user preferences
 - ***If user has not opted in***:
   - Abort
 - ***else***:
   - retrieve remote manifest from update_ver.json [manifest](https://github.com/T3RRYT3RR0R/Image-Sorter/blob/main/updates/update_ver.json)
   - iterate remote manifest
   - test if manifest item exists on local path, ***if false***:
     - download item and continue
       ***else***:
     - download item if the version is newer than that recorded in the local manifest.

### A note on the methodology of Base-code / Update handling:
code within the script is segmented and wrapped in a function call that:
 - If no corresponding update file exists, executes the base code
  - ***else:***
 - Executes the code in the update file.
 - Update segments may have cross dependencies. For this reason,
   updates from the manifest are persistent.
   Commits of updates will note if there is cross dependencies
   in the form:
   update segments segID+segID...
