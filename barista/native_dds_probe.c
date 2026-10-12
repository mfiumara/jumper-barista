/* Link and load CycloneDDS without opening a participant or robot bus. */
#include <dds/dds.h>
#include <stdio.h>

int main(void)
{
    dds_qos_t *qos = dds_create_qos();
    if (!qos) return 1;
    dds_delete_qos(qos);
    puts("CycloneDDS native link/load and QoS allocation succeeded (no participant).");
    return 0;
}
