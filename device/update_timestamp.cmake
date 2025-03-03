# Helper script to update timestamp in index.html
file(READ "/home/joe/Projects/SEAL-Embedded/device/build/reports/index.html.in" index_content)
string(REPLACE "@TIMESTAMP@" "${TIMESTAMP}" updated_content "${index_content}")
file(WRITE "/home/joe/Projects/SEAL-Embedded/device/build/reports/index.html" "${updated_content}")
