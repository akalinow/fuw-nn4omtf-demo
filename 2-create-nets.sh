#!/bin/bash

set -o errexit

. paths.sh

if [ -e "${NETS}" ]; then
  echo "Deleting old models..."
  rm -fr ${NETS}
fi

omtfnntool storage ${REDUCED_STORE} ${NETS}

omtfnntool network r1 ${BUILDERS}/red_1.py ${NETS}/${REDUCED_STORE}
omtfnntool network r2 ${BUILDERS}/red_2.py ${NETS}/${REDUCED_STORE}
