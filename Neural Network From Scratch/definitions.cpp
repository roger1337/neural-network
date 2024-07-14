
#include "definitions.h"

dt threshold(dt in) {
	if (in > -1e-250 && in < 1e-250)
		return 0;
	return in;
}