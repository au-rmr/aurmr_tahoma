uint16[16] SETPOINT_H1 # UINT16 Length 16 x 2 rw
uint16[16] HYSTERESIS_h1 # UINT16 Lenth 16 x 2 rw
uint16[16] SETPOINT_H2 # UINT16 Lenth 16 x 2 rw
uint16[16] HYSTERESIS_h2 # UINT16 Lenth 16 x 2 rw

uint16[16] SYSTEM_VACUUM # UINT16 Length 16 x 2 ro
uint16[16] LEAKAGE_RATE # UINT16 Length 16 x 2 ro
uint16[16] FREE_FLOW_VACUUM # UINT16 Length 16 x 2 ro
uint16[16] MAX_VACUUM_REACHED # UINT16 Length 16 x 2 ro

int8[16] EJECTOR_CONTROL # UINT8 Length 16 rw

float32 SUPPLY_VOLTAGE # float32 length 1 byte ro
float32 SUPPLY_PRESSURE # float32 Length 1 byte rw