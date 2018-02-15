#!/bin/bash

. paths.sh

omtfnntool storage ${FULL_STORE} ${NETS}

omtfnntool storage ${REDUCED_STORE} ${NETS}

omtfnntool network ex ${BUILDERS}/exp.py ${NETS}
omtfnntool network r1 ${BUILDERS}/red_1.py ${NETS}/${REDUCED_STORE}
omtfnntool network r2 ${BUILDERS}/red_2.py ${NETS}/${REDUCED_STORE}
omtfnntool network r3 ${BUILDERS}/red_3.py ${NETS}/${REDUCED_STORE}
omtfnntool network f1 ${BUILDERS}/full_1.py ${NETS}/${FULL_STORE}
omtfnntool network f2 ${BUILDERS}/full_2.py ${NETS}/${FULL_STORE}
omtfnntool network f3 ${BUILDERS}/full_3.py ${NETS}/${FULL_STORE}
