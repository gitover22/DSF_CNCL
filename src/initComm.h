#ifndef INITCOMM_H
#define INITCOMM_H

#include "cncl.h"
#include "cnrt.h"
#include "mluTool.h"
#include "cnclComm.h"
/**
 * @brief Initialize the CNCL communicator
 * @return true if the CNCL communicator is initialized successfully, false otherwise
*/
bool InitComm(cnclComm_t *comms, int num_comm, \
    const int *dev_list, const int *rank_list, int nrank, cnclCliqueId_t clique_id);


/**
 * @brief Map the CNCL communicator to CNCLComm object
*/
bool Map_Comm(cnclComm_t *tComm_list,CNCLComm* comm_list,int comms);
#endif