
#include "process_group_cncl.hpp"

#include <map>

#ifdef TEST_COVERAGE
extern "C" void __gcov_flush();
#endif

namespace torch_mlu {

namespace {

// RAII helper class to manage CNCL group API and CUDA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
/**
 * RAII助手类来管理CNCL组API和CUDA自由互斥,析构函数允许抛出，因为这个辅助类只管理组和锁的生命周期。
*/
struct AutoCnclGroup {
  AutoCnclGroup() {
    (torch_mlu::getFreeMutex())->lock();
    C10D_CNCL_CHECK(cnclGroupStart());
  }
  ~AutoCnclGroup() noexcept(false) {
    C10D_CNCL_CHECK(cnclGroupEnd());  // NOSONAR
    (torch_mlu::getFreeMutex())->unlock();
  }
};

// CNCL 规约操作类型 map存储映射关系
const std::map<c10d::ReduceOp, cnclReduceOp_t> cncl_op = {
    {c10d::ReduceOp::MIN, cnclMin},
    {c10d::ReduceOp::MAX, cnclMax},
    {c10d::ReduceOp::SUM, cnclSum},
    {c10d::ReduceOp::PRODUCT, cnclProd},
};

// CNCL 数据类型 
std::map<at::ScalarType, cnclDataType_t> cncl_data_type = {
    {at::kChar, cnclInt8}, {at::kByte, cnclUint8}, {at::kFloat, cnclFloat},
    {at::kInt, cnclInt32}, {at::kLong, cnclInt32}, {at::kHalf, cnclHalf},
    {at::kDouble, cnclFloat}, {at::kBool, cnclUint8}, {at::kBFloat16, cnclBfloat16}
};

// 一个用于获取 CNCL 数据类型的帮助函数，如果传入的数据类型不受支持，则会抛出异常
cnclDataType_t getCnclDataType(at::ScalarType type) {
  try {
    return cncl_data_type.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for CNCL process group");
  }
}
// c10d::ReduceOp 枚举类型和输入张量的数据类型，获取对应的 CNCL 规约操作类型
cnclReduceOp_t getCnclReduceOp(const c10d::ReduceOp reduce_op, at::Tensor& input) {
  try {
    if (reduce_op == c10d::ReduceOp::SUM && input.scalar_type() == at::kBool) {
      // For bool tensors, map sum to max, which both represent a bitwise or.
      // This is to prevent overflow issues with sum, since we use uint8 to
      // represent a bool (see cnclDataType mapping).
      return cnclMax;
    }
    return cncl_op.at(reduce_op);
  } catch (const std::out_of_range& e) {
    switch (reduce_op) {
      case c10d::ReduceOp::BAND:
        throw std::runtime_error("Cannot use ReduceOp.BAND with CNCL");
        break;
      case c10d::ReduceOp::BOR:
        throw std::runtime_error("Cannot use ReduceOp.BOR with CNCL");
        break;
      case c10d::ReduceOp::BXOR:
        throw std::runtime_error("Cannot use ReduceOp.BXOR with CNCL");
        break;
      default:
        throw std::runtime_error("Unhandled ReduceOp");
        break;
    }
  }
}

// 从设备列表中获取设备列表字符串
// 接收一个设备（at::Device）的vector向量作为参数，并返回第一个设备的索引作为字符串
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  return std::to_string(devices[0].index());
}

/**
 * @brief 接收两个整型参数my_rank和peer，分别表示两个参与通信的rank号 \
 * 它返回一个字符串，该字符串以“低排名:高排名”的格式表示发送和接收对 \
 * 例如，如果my_rank是2，peer是5，那么返回的字符串将是“2:5”
*/
std::string getKeySendRecv(int my_rank, int peer) {
  int low_rank = my_rank < peer ? my_rank : peer;
  int high_rank = my_rank < peer ? peer : my_rank;
  std::string send_recv_pair =
      std::to_string(low_rank) + ":" + std::to_string(high_rank);
  return send_recv_pair;
}

// 从张量列表中获取设备列表
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size()); // 预留足够的空间以容纳所有张量的设备
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

// [Sync Queues] Helper that lets the input cncl_queue to wait for the current
// queue. CNCL communications run on cncl_queue, but input tensors are
// allocated on different queues (i.e., current queues). Communications on
// cncl_queue cannot start before pending input tensor ops on current queues
// finish. Otherwise, ops on two queues might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on cncl_queue finish. This
// can be achieved by calling torch_mlu::recordQueue,
// which remembers the usage queue (cncl_queue), creates an notifier on the usage
// queue when GC attempts to free the input tensor, and delays GC until that
// notifier is done.
/**
 * @brief 同步队列
 * 使得输入的 cncl_queue 等待当前队列上的操作完成
 * 在当前队列上的待处理输入张量操作完成之前，cncl_queue上的通信不能开始。否则，两队列上的操作可能会同时读取/写入相同的张量。
*/
void syncQueues(const at::Device& device, std::vector<torch_mlu::Notifier>& cncl_notifiers,
                torch_mlu::Queue& cncl_queue) {
  auto current_queue = torch_mlu::getCurrentQueue(device.index());
  cncl_notifiers[0].place(current_queue);
  cncl_notifiers[0].wait(cncl_queue);
  if (torch_mlu::PythonInterface::getAsyncMode() == false) {
    current_queue.synchronize();
  }
}

}  // namespace

thread_local uint64_t ProcessGroupCNCL::cnclActiveGroupCounter_ = 0;

ProcessGroupCNCL::WorkCNCL::WorkCNCL(const std::vector<at::Device>& devices)
    : devices_(devices) {
  cncl_end_notifiers_ =  
      std::make_shared<std::vector<torch_mlu::Notifier>>(
          devices.size()); 
}

ProcessGroupCNCL::WorkCNCL::~WorkCNCL() {}

// currently MLU do not support CNCL error check
bool ProcessGroupCNCL::WorkCNCL::isCompleted() { return finishedMLUExecutionInternal(); }

// currently MLU do not support CNCL error check
bool ProcessGroupCNCL::WorkCNCL::isSuccess() const { return finishedMLUExecutionInternal(); }

bool ProcessGroupCNCL::WorkCNCL::finishedMLUExecutionInternal() const {
  // 检查工作的对应MLU通知器的状态
  try {
    for (const auto i : c10::irange(devices_.size())) {
      // 检查工作的对应CUDA事件的状态
      if (!(*cncl_end_notifiers_)[i].query()) {
        return false;
      }
    }
  } catch (const std::exception& e) {
    // 如果发生异常且异常消息不包含“driver shutting down”
    if (std::string(e.what()).find("driver shutting down") ==
        std::string::npos) {
      throw;// 重新抛出异常
    }
    LOG(INFO) << "[Rank " << rank_
              << "] Event query failed with exception: " << e.what();
  }
  return true;// 如果所有通知器都已完成，返回true
}

std::vector<at::Tensor> ProcessGroupCNCL::WorkCNCL::result() {
  return *outputs_;
}

void ProcessGroupCNCL::WorkCNCL::synchronizeQueues() {
  for (const auto i : c10::irange(devices_.size())) {
    auto current_queue = getCurrentQueue(devices_[i].index());
    // 阻塞当前流上的NCCL流
    (*cncl_end_notifiers_)[i].wait(current_queue);
  }
}

// 等待工作的对应CNRT事件
void ProcessGroupCNCL::WorkCNCL::synchronize() {

  synchronizeQueues();

  auto current_queue = torch_mlu::getCurrentQueue(devices_[0].index());
  
  if (blockingWait_) {
    current_queue.synchronize(); // 如果是阻塞等待，进行同步
  }

  if (!barrier_tensors_.empty()) {
    torch_mlu::mlu::MLUGuard guard(devices_[0]);
    TORCH_CNRT_CHECK(cnrtSyncDevice());
  }

  return;
}

// Same as calling synchronize().
bool ProcessGroupCNCL::WorkCNCL::wait(std::chrono::milliseconds timeout) {
  synchronize();
  return true;
}

ProcessGroupCNCL::ProcessGroupCNCL(
    const c10::intrusive_ptr<c10d::Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : ProcessGroup(rank, size), store_(store) {
  TORCH_CHECK(
      torch_mlu::device_count() != 0,
      "ProcessGroupCNCL is only supported with MLUs, no MLUs found!");
  char* blockingWait = getenv(CNCL_BLOCKING_WAIT);
  try {
    if (blockingWait != nullptr) {
      auto val = std::stoi(blockingWait);
      if (val == 1) {
        // Make wait() and synchronize() a blocking call.
        blockingWait_ = true;
      } else if (val != 0) {
        throw std::runtime_error(
            "Invalid value for environment variable: " +
            std::string(CNCL_BLOCKING_WAIT));
      }
    }
  } catch (std::exception& e) {
    throw std::runtime_error(
        "Invalid value for environment variable: " +
        std::string(CNCL_BLOCKING_WAIT));
  }
}

ProcessGroupCNCL::~ProcessGroupCNCL(){
  {
    // Abort all CNCL Communicators on Process Group Destruction
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : dev_cncl_comm_map_) {
      auto& cncl_comms = it.second;

      for (const auto& cncl_comm : cncl_comms) {
        cncl_comm->cnclCommAbort();
      }
    }
  }
// gcov can not save the coverage data of the code run by subprocess,
// so we flush the coverge data manually
#ifdef TEST_COVERAGE
  __gcov_flush();
#endif
}

void ProcessGroupCNCL::broadcastCNCLCliqueID(
    cnclCliqueId* cncl_id,
    const bool is_p2p_op = false,
    const std::string& p2p_key = "",
    const int p2p_rank = 0) {
  // For collective operations:
  // For every CNCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple CNCL communicators, so we use a sequence
  // number to differentiate between them.
  // For point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string store_key;
  if (!is_p2p_op) {
    store_key = std::to_string(cncl_comm_counter_++);
  } else {
    store_key  = p2p_key;
  }
  if (rank_ == 0 || (is_p2p_op && p2p_rank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(cncl_id),
        reinterpret_cast<uint8_t*>(cncl_id) + CNCL_CLIQUE_ID_BYTES_SIZE);
    store_->set(store_key, vec);
  } else {
    auto vec = store_->get(store_key);
    if (vec.size() != CNCL_CLIQUE_ID_BYTES_SIZE) {
      throw std::runtime_error(
          "Unexpected CNCL clique ID length received "
          "from the store");
    }
    std::memcpy(cncl_id, vec.data(), vec.size());
  }
}

std::vector<std::shared_ptr<CNCLComm>>& ProcessGroupCNCL::getCNCLComm(
    const std::string& devices_key,
    const std::vector<at::Device>& devices,
    c10d::OpType op_type,
    const int p2p_rank,
    const bool is_send_recv_self) {
  // 设备为空抛出异常
  if (devices_key.empty()) {
    throw std::runtime_error(
        "Not able to create/get the CNCL Communicator since "
        "the MLU devices are not known");
  }
  // 添加设备索引index到usedDeviceIdxs
  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    if (dev_cncl_comm_map_.find(devices_key) != dev_cncl_comm_map_.end()) {
      // 如果有缓存的通信子，则复用
      return dev_cncl_comm_map_[devices_key];
    }
  }

  // 如果没有缓存的通信子，则创建一个新的 CNCL 通信器向量 cncl_comms
  std::vector<std::shared_ptr<CNCLComm>> cncl_comms;
  cncl_comms.resize(devices.size());

  // For batch_isend_irecv, cnclGroupStart() would be called upfront
  bool batch_p2p = cnclActiveGroupCounter_ > 0;
  bool single_p2p_op = c10d::isP2POp(op_type, batch_p2p);

  // Create the unique CNCL ID and broadcast it
  cnclCliqueId clique_id;

  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (single_p2p_op && p2p_rank == 0)) {
    C10D_CNCL_CHECK(cnclGetCliqueId(&clique_id));
  }

  // For point-to-point communication on the same process, don't need broadcast.
  if (!is_send_recv_self) {
    // Broadcast so that each process can have a unique CNCL ID
    broadcastCNCLCliqueID(&clique_id, single_p2p_op, devices_key, p2p_rank);
  }

  std::vector<torch_mlu::Queue> queue_val;
  queue_val.reserve(devices.size());

  // [Group Start/End Note] This is used to ensure that cncl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying CNCL calls will look like
  //   cnclGroupStart() // This is in batch_isend_irecv
  //   cnclGroupStart() // This is [Note 1]
  //   cnclInitComms() // Inside CNCLComm::create
  //   cnclSend()
  //   cnclGroupEnd() // This is [Note 2]
  //   cnclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the cncl communicator will be created in the last
  // cnclGroupEnd which means when cnclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active cncl groups to ensure cncl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (size_t i = 0; i < cnclActiveGroupCounter_; ++i) {
    C10D_CNCL_CHECK(cnclGroupEnd());
  }

  // [Note 1] Create the CNCL communicators for each MLU
  C10D_CNCL_CHECK(cnclGroupStart());

  for (const auto i : c10::irange(devices.size())) {
    // world size and rank
    int num_ranks, rank_id;

    if (!single_p2p_op) {
      // Collective, all-to-all, or batch P2P
      // One rank for each device
      num_ranks = getSize() * devices.size();
      rank_id = getRank() * devices.size() + i;
    } else if (is_send_recv_self) {
      // Same process send and recv.
      num_ranks = 1;
      rank_id = 0;
    } else {
      // For point-to-point operation, there are only 2 processes involved so
      // the MLU rank is either 0 or 1.
      num_ranks = 2;
      rank_id = p2p_rank;
    }

    // Create the CNCL communicators for each MLU
    cncl_comms[i] =
        CNCLComm::create(num_ranks, rank_id, devices[i].index(), &clique_id);

    // Create streams
    queue_val.push_back(torch_mlu::getQueueFromPool(
        false /*options_->is_high_priority_stream*/, devices[i].index()));
  }

  // [Note 2 ]
  C10D_CNCL_CHECK(cnclGroupEnd());

  // See [Group Start/End Note]
  for (size_t i = 0; i < cnclActiveGroupCounter_; ++i) {
    C10D_CNCL_CHECK(cnclGroupStart());
  }

  cncl_queues_.emplace(devices_key, std::move(queue_val));

  cncl_notifiers_.emplace(std::piecewise_construct,
                          std::make_tuple(devices_key),
                          std::make_tuple(devices.size()));

  // Move the CNCL resource to cache
  dev_cncl_comm_map_.emplace(devices_key, std::move(cncl_comms));
  return dev_cncl_comm_map_[devices_key];
}

namespace {

// Check validity of tensor
void check_mlu_single_tensor(const at::Tensor& tensor) {
  if (!tensor.is_mlu() || tensor.is_sparse()) {
    throw std::runtime_error("Tensors must be MLU and dense");
  }
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    throw std::runtime_error("Tensors must be contiguous");
  }
}

// Check that all `tensors' have the same type and shape and are distributed
// across distinct MLUs.
void check_mlu_tensors_different_devices(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(torch_mlu::device_count())) {
    throw std::runtime_error(
        "Tensor list mustn't be larger than the number of available MLUs");
  }

  if (tensors.size() != 1) {
    throw std::runtime_error(
        "MLU Tensors must be on a single MLU device per process");
  }

  const auto& first = tensors.front();
  if (!first.is_mlu() || first.is_sparse()) {
    throw std::runtime_error("Tensors must be MLU and dense");
  }

  if (!first.is_contiguous(first.suggest_memory_format())) {
    TORCH_CHECK(false, "Tensors must be contiguous");
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_tensor_list(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other, size_t world_size) {
  if (tensor_lists.size() != 1 || other.size() != 1) {
    throw std::runtime_error(
        "MLU Tensors must be on a single MLU device per process");
  }

  if (tensor_lists[0].size() == 0) {
    throw std::runtime_error("Received an empty list");
  }

  if (tensor_lists[0].size() != world_size) {
    throw std::runtime_error(
        "Tensor list input to scatter/gather must match number of collective"
        " participants");
  }

  auto device = other[0].device();
  for (const auto& t : tensor_lists[0]) {
    if (t.numel() != other[0].numel()) {
      throw std::runtime_error(
          "All tensor operands to scatter/gather must have the same number of elements");
    }
    if (t.device() != device) {
      throw std::runtime_error("Expecting all tensors on the same device");
    }
  }

  auto& t = tensor_lists[0][0];
  std::vector<int64_t> new_size{static_cast<int64_t>(tensor_lists[0].size())};
  std::vector<int64_t> new_stride{t.numel()};
  new_size.insert(new_size.end(), t.sizes().begin(), t.sizes().end());
  new_stride.insert(new_stride.end(), t.strides().begin(), t.strides().end());
  return {at::empty_strided(new_size, new_stride, t.options().memory_format(c10::nullopt))};
}

}  // namespace

template <typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    c10d::OpType op_type
    ) {
  const auto devices = getDeviceList(inputs);
  const auto key = getKeyFromDevices(devices);

  auto& cncl_comms = getCNCLComm(key, devices, op_type);
  torch_mlu::Queue& cncl_queue = cncl_queues_[key][0];
  // First let CNCL queue wait for input tensors allocation queue
  syncQueues(devices[0], cncl_notifiers_[key], cncl_queue);

  if (coalescing_active_) {
    coalescedDevices_.push_back(devices);
  }

  // Work itself will create the CNCL notifiers on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  // Store references to outputs to be used by WorkCNCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  // Both `inputs' and `outputs' are created on a worker queue and used in
  // different cncl queue.  Hence, both must record the cncl queue to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Queues].
  {
    AutoCnclGroup cncl_group_guard;
    torch_mlu::mlu::MLUGuard guard(devices[0]);
    torch_mlu::recordQueue(inputs[0].storage().data_ptr(), cncl_queue);
    fn(inputs[0], outputs[0], cncl_comms[0]->getCnclComm(), cncl_queue);
  }

  // End event should only be recorded after the cnclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    auto& cncl_queue = cncl_queues_[key][i];
    if (!coalescing_active_) {
      (*work->cncl_end_notifiers_)[i].place(cncl_queue);
    }
  }


  {  
    // Set current stream to init future's event with cncl queue
    torch_mlu::mlu::MLUQueueGuard queue_guard(cncl_queue);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  work->blockingWait_ = blockingWait_;

  return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  check_mlu_tensors_different_devices(tensors);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        C10D_CNCL_CHECK(cnclAllReduce(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            getCnclReduceOp(opts.reduceOp, input),
            comm,
            queue.queue()));
        return;
      },
      c10d::OpType::ALLREDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  check_mlu_tensors_different_devices(tensors);

  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        const int root = opts.rootRank * tensors.size() + opts.rootTensor;
        C10D_CNCL_CHECK(cnclBroadcast(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            root,
            comm,
            queue.queue()));
        return;
      },
      c10d::OpType::BROADCAST);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
  check_mlu_tensors_different_devices(tensors);

  return collective(
      tensors, tensors,
      [&](at::Tensor& input, at::Tensor& output, cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        const int root = opts.rootRank * tensors.size() + opts.rootTensor;
        C10D_CNCL_CHECK(cnclReduce(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            getCnclReduceOp(opts.reduceOp, input),
            root,
            comm,
            queue.queue()));
        return;
      },
      c10d::OpType::REDUCE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions& opts) {
  check_mlu_tensors_different_devices(input_tensors);
  auto output_flattened = flatten_tensor_list(output_tensors, input_tensors, size_);

  return collective(
      input_tensors, output_flattened,
      [&](at::Tensor& input, at::Tensor& output, cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();

        torch_mlu::recordQueue(output.storage().data_ptr(), queue);
        C10D_CNCL_CHECK(cnclAllGather(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            comm,
            queue.queue()));

        torch_mlu::mlu::MLUQueueGuard guard(queue);
        for (size_t i = 0; i < output_tensors[0].size(); ++i) {
          torch_mlu::recordQueue(
              output_tensors[0][i].storage().data_ptr(), queue);
          output_tensors[0][i].copy_(output[i], true);
        }
        return;
      },
      c10d::OpType::ALLGATHER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10d::AllgatherOptions& opts) {
    check_mlu_single_tensor(input_tensor);
    check_mlu_single_tensor(output_tensor);

  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * size_ != output_tensor.numel()) {
    TORCH_CHECK(
        false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor> {input_tensor};
  auto outputs = std::vector<at::Tensor> {output_tensor};

  return collective(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        torch_mlu::recordQueue(output.storage().data_ptr(), queue);
        C10D_CNCL_CHECK(cnclAllGather(
            input_ptr,
            output_ptr,
            input.numel(),
            getCnclDataType(input.scalar_type()),
            comm,
            queue.queue()));
        return;
      },
      c10d::OpType::_ALLGATHER_BASE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const c10d::AllgatherOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
    check_mlu_tensors_different_devices(output_tensors);
    auto input_flattended =
        flatten_tensor_list(input_tensors, output_tensors, size_);
    check_mlu_tensors_different_devices(input_flattended);

    return collective(
        input_flattended, output_tensors,
        [&](at::Tensor& input, at::Tensor& output, cnclComm_t comm,
            torch_mlu::Queue& queue) {
          torch_mlu::mlu::MLUQueueGuard guard(queue);
          for (size_t i = 0; i < input_tensors[0].size(); ++i) {
            torch_mlu::recordQueue(input_tensors[0][i].storage().data_ptr(),
                                   queue);
            input_flattended[0][i].copy_(input_tensors[0][i], true);
          }
          auto input_impl = torch_mlu::getMluTensorImpl(input);
          auto input_ptr = input_impl->mlu_data_ptr();
          auto output_impl = torch_mlu::getMluTensorImpl(output);
          auto output_ptr = output_impl->mlu_data_ptr();

          torch_mlu::recordQueue(output.storage().data_ptr(), queue);
          C10D_CNCL_CHECK(cnclReduceScatter(
              input_ptr, output_ptr, output.numel(),
              getCnclDataType(input.scalar_type()),
              getCnclReduceOp(opts.reduceOp, input), comm, queue.queue()));
          return;
        },
        c10d::OpType::REDUCE_SCATTER);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::_reduce_scatter_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10d::ReduceScatterOptions& opts) {
  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(
        false, "input tensor must be the same type as the output tensor.");
  }

  if (input_tensor.numel() != output_tensor.numel() * size_) {
    TORCH_CHECK(
        false,
        "input tensor must be the same size as output size times world size.");
  }

  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};

  return collective(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        torch_mlu::recordQueue(output.storage().data_ptr(), queue);
        C10D_CNCL_CHECK(cnclReduceScatter(
            input_ptr,
            output_ptr,
            output.numel(),
            getCnclDataType(input.scalar_type()),
            getCnclReduceOp(opts.reduceOp, input),
            comm,
            queue.queue()));
        return;
      },
      c10d::OpType::_REDUCE_SCATTER_BASE);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::barrier(
    const c10d::BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a CNCL collective being called
    // Here we have to use the best guesses and will use a single MLU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different MLU
    auto num_mlus = torch_mlu::device_count();
    int16_t device_idx = static_cast<int16_t>(rank_ % num_mlus);
      LOG(INFO) << c10::str(
      "Rank ",
      this->getRank(),
      " using MLU ",
      device_idx,
      " to perform barrier as devices used by this process are currently unknown. ",
      "This can potentially cause a hang if this rank to MLU mapping is incorrect.",
      "Specify device_ids in barrier() to force use of a particular device.");
    devices.push_back(at::Device(at::DeviceType::MLU, device_idx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(at::DeviceType::MLU, usedDeviceIdx));
    }
  }

  std::vector<at::Tensor> barrier_tensors;
  barrier_tensors.reserve(devices.size());

  torch_mlu::mlu::OptionalMLUGuard mlu_guard;
  for (auto& device : devices) {
    mlu_guard.set_index(device.index());
    barrier_tensors.push_back(at::empty({1},
      at::TensorOptions().device(at::DeviceType::MLU).dtype(at::kFloat)));
  }

  // All reduce to achieve the barrier
  auto work = allreduce(barrier_tensors);

  // Work will take over barrierTensors
  auto cncl_work = dynamic_cast<ProcessGroupCNCL::WorkCNCL*>(work.get());
  TORCH_CHECK(cncl_work);
  cncl_work->barrier_tensors_ = std::move(barrier_tensors);

  return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::alltoall_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes,
    const c10d::AllToAllOptions& /* unused */) {
  check_mlu_single_tensor(output_tensor);
  check_mlu_single_tensor(input_tensor);
  std::vector<at::Tensor> input_tensors = {input_tensor};
  std::vector<at::Tensor> output_tensors = {output_tensor};

  return collective(
      input_tensors, output_tensors,
      [&](at::Tensor& input, at::Tensor& output, cnclComm_t comm,
          torch_mlu::Queue& queue) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        auto input_ptr = input_impl->mlu_data_ptr();
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        auto output_ptr = output_impl->mlu_data_ptr();
        auto dtype = getCnclDataType(input.scalar_type());

        // When equal split, use cnclAlltoAll to improve performance
        if (output_split_sizes.size() == 0 && input_split_sizes.size() == 0) {
          int64_t cnt = input.numel() / size_;
          if (cnt == 0) {
            return;
          }
          torch_mlu::recordQueue(output.storage().data_ptr(), queue);
          C10D_CNCL_CHECK(cnclAlltoAll(
              input_ptr,
              output_ptr,
              cnt,
              dtype,
              comm,
              queue.queue()));
        } else {
          c10d::checkSplitSizes(input_split_sizes, input, size_);
          c10d::checkSplitSizes(output_split_sizes, output, size_);
          std::vector<const void*> send_buffers(size_);
          std::vector<uint32_t> send_counts(size_);
          std::vector<void*> recv_buffers(size_);
          std::vector<uint32_t> recv_counts(size_);
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          std::vector<cnclDataType_t> send_types(size_, dtype);
          std::vector<cnclDataType_t> recv_types(size_, dtype);
          auto itemsize = torch_mlu::getCnnlTypeSize(torch_mlu::getCnnlType(input_impl));
          c10d::computeLengthsAndOffsets(
              input_split_sizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              output_split_sizes, output, &recv_lengths, &recv_offsets);
          bool empty_send = true;
          for (int64_t r = 0; r < size_; ++r) {
            if (empty_send && send_lengths[r] > 0) {
              empty_send = false;
            }
            send_buffers[r] =
                    static_cast<const void*>(static_cast<char*>(input_ptr) +
                                             send_offsets[r] * itemsize);
            send_counts[r] = c10::checked_convert<uint32_t, size_t>(send_lengths[r], "uint32_t");
            recv_buffers[r] =
                static_cast<void*>(static_cast<char*>(output_ptr) + recv_offsets[r] * itemsize);
            recv_counts[r] = c10::checked_convert<uint32_t, size_t>(recv_lengths[r], "uint32_t");
          }
          if (empty_send) {
            return;
          }
          torch_mlu::recordQueue(output.storage().data_ptr(), queue);
          C10D_CNCL_CHECK(cnclAlltoAllv(
              send_buffers.data(),
              send_counts.data(),
              send_types.data(),
              recv_buffers.data(),
              recv_counts.data(),
              recv_types.data(),
              comm,
              queue.queue()));
        }
        return;
      },
      c10d::OpType::ALLTOALL_BASE);
}

ProcessGroupCNCL::Options::Options(bool is_high_priority_stream)
    : ProcessGroup::Options(CNCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions& /* unused */) {
  TORCH_MLU_CHECK(input_tensors.size() == size_ && output_tensors.size() == size_,
      "Size of input tensor list not equal group size");
  auto device = output_tensors[0].device();
  for (size_t r = 0; r < output_tensors.size(); r++) {
    check_mlu_single_tensor(output_tensors[r]);
    check_mlu_single_tensor(input_tensors[r]);
    TORCH_MLU_CHECK(device == output_tensors[r].device() && device == input_tensors[r].device(),
        "Tensors must be on the same device");
  }

  std::vector<at::Tensor> input_tensor0 = {input_tensors[0]};
  std::vector<at::Tensor> output_tensor0 = {output_tensors[0]};
  return collective(
      input_tensor0, output_tensor0,
      [&](at::Tensor& /* unused */, at::Tensor& /* unused */, cnclComm_t comm,
          torch_mlu::Queue& queue) {
        std::vector<const void*> send_buffers(size_);
        std::vector<uint32_t> send_counts(size_);
        std::vector<cnclDataType_t> send_types(size_);
        std::vector<void*> recv_buffers(size_);
        std::vector<uint32_t> recv_counts(size_);
        std::vector<cnclDataType_t> recv_types(size_);
        bool empty_send = true;
        for (size_t r = 0; r < size_; ++r) {
          auto input_impl = torch_mlu::getMluTensorImpl(input_tensors[r]);
          send_buffers[r] = static_cast<const void*>(input_impl->mlu_data_ptr());
          send_counts[r] =
              c10::checked_convert<uint32_t, int64_t>(input_impl->numel(), "uint32_t");
          send_types[r] = getCnclDataType(input_tensors[r].scalar_type());
          auto output_impl = torch_mlu::getMluTensorImpl(output_tensors[r]);
          recv_buffers[r] = output_impl->mlu_data_ptr();
          recv_counts[r] =
              c10::checked_convert<uint32_t, int64_t>(output_impl->numel(), "uint32_t");
          recv_types[r] = getCnclDataType(output_tensors[r].scalar_type());
          if (send_counts[r] > 0) {
            empty_send = false;
          }
        }
        if (empty_send) {
          return;
        }
        // input_tensors[0] have already been recorded in collective functions.
        for (size_t r = 1; r < size_; ++r) {
          torch_mlu::recordQueue(input_tensors[r].storage().data_ptr(), queue);
        }
        C10D_CNCL_CHECK(cnclAlltoAllv(
            send_buffers.data(),
            send_counts.data(),
            send_types.data(),
            recv_buffers.data(),
            recv_counts.data(),
            recv_types.data(),
            comm,
            queue.queue()));
        return;
      },
      c10d::OpType::ALLTOALL);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */, const c10d::GatherOptions& /* unused */) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const c10d::ScatterOptions& /* unused */) {
  throw std::runtime_error("Not supported yet");
}

void ProcessGroupCNCL::groupStart() {
  C10D_CNCL_CHECK(cnclGroupStart());
  ++cnclActiveGroupCounter_;
}

void ProcessGroupCNCL::groupEnd() {
  C10D_CNCL_CHECK(cnclGroupEnd());
  --cnclActiveGroupCounter_;
}

template <typename Fn>
    c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::pointToPoint(
        std::vector<at::Tensor>& tensors, Fn fn, int peer, c10d::OpType op_type,
        const char* profilingTitle) {
  std::string key;
  const auto devices = getDeviceList(tensors);
  const bool is_p2p_op = c10d::isP2POp(op_type);

  int p2p_rank = 0, p2p_target_rank = 0;
  bool is_send_recv_self = false;

  // For batch_isend_irecv, cnclGroupStart() would be called upfront
  bool batchP2P = cnclActiveGroupCounter_ > 0;

  if (batchP2P) {
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = getKeyFromDevices(devices);
    p2p_rank = rank_;
    p2p_target_rank = peer;
  } else {
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = getKeySendRecv(rank_, peer);
    p2p_rank = rank_ <= peer ? 0 : 1;
    is_send_recv_self = rank_ == peer;
    p2p_target_rank = is_send_recv_self ? 0 : 1 - p2p_rank;
  }

  auto& cncl_comms =
      getCNCLComm(key, devices, op_type, p2p_rank, is_send_recv_self);

  if (coalescing_active_) {
    coalescedDevices_.push_back(devices);
  }

  torch_mlu::Queue& cncl_queue = cncl_queues_[key][0];
  syncQueues(devices[0], cncl_notifiers_[key], cncl_queue);

  // Work itself will create the CNCL events on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(tensors);

  {
    AutoCnclGroup cncl_group_guard;
    torch_mlu::mlu::MLUGuard guard(devices[0]);
    torch_mlu::recordQueue(tensors[0].storage().data_ptr(), cncl_queue);
    C10D_CNCL_CHECK(
        fn(tensors[0], cncl_comms[0]->getCnclComm(), cncl_queue, p2p_target_rank));
  }

   // End event should only be recorded after the cnclGroupEnd()
  for (const auto i : c10::irange(devices.size())) {
    auto& cncl_queue = cncl_queues_[key][i];
    if (!coalescing_active_) {
      (*work->cncl_end_notifiers_)[i].place(cncl_queue);
    }
  }

  {
    // Set current stream to init future's event with cncl queue
    torch_mlu::mlu::MLUQueueGuard queue_guard(cncl_queue);
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    // Set current stream to init future's event with cncl queue
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }
  
  if (work->recordFunctionEndCallback_) {
    work->future_->addCallback([work](at::ivalue::Future& /* unused */) {
      work->recordFunctionEndCallback_();
    });
  }

  work->blockingWait_ = blockingWait_;

  return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::send(
    std::vector<at::Tensor>& tensors, int dst_rank, int /* unused */) {
  check_mlu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& input, cnclComm_t comm, torch_mlu::Queue& queue,
          int dst) {
        auto input_impl = torch_mlu::getMluTensorImpl(input);
        return cnclSend(
            input_impl->mlu_data_ptr(),
            input.numel(),
            getCnclDataType(input.scalar_type()), dst, comm, queue.queue());
      },
      dst_rank, c10d::OpType::SEND, "cncl:send");
  return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::recv(
    std::vector<at::Tensor>& tensors, int src_rank, int /* unused */) {
  check_mlu_tensors_different_devices(tensors);
  auto ret = pointToPoint(
      tensors,
      [&](at::Tensor& output, cnclComm_t comm, torch_mlu::Queue& queue,
          int src) {
        auto output_impl = torch_mlu::getMluTensorImpl(output);
        return cnclRecv(
            output_impl->mlu_data_ptr(),
            output.numel(),
            getCnclDataType(output.scalar_type()), src, comm, queue.queue());
      },
      src_rank, c10d::OpType::RECV, "cncl:recv");
  return ret;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupCNCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */, int /* unused */) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupCNCL::WorkCNCL::
    getFuture() {
  return future_;
}

void ProcessGroupCNCL::startCoalescing() {
  coalescedDevices_.clear();
  coalescing_active_ = true;
  groupStart();
}

void ProcessGroupCNCL::endCoalescing(
    std::vector<c10::intrusive_ptr<c10d::Work>>& reqs) {
  groupEnd();
  if (reqs.size() != coalescedDevices_.size()) {
    TORCH_CHECK(false, "Number of requests do not match number of collectives");
  }

  int batch_idx = 0;
  for (const auto& req : reqs) {
    auto cncl_work = static_cast<ProcessGroupCNCL::WorkCNCL*>(req.get());
    // @lint-ignore CLANGTIDY
    std::vector<at::Device> devices = coalescedDevices_[batch_idx];
    const auto key = getKeyFromDevices(devices);
    auto cncl_queues = cncl_queues_[key];
    for (const auto i : c10::irange(devices.size())) {
      (*cncl_work->cncl_end_notifiers_)[i].place(cncl_queues[i]);
    }
    batch_idx += 1;
  }
  coalescing_active_ = false;
}

c10::intrusive_ptr<c10d::ProcessGroup> ProcessGroupCNCL::createProcessGroupCNCL(
    const c10::intrusive_ptr<::c10d::Store> &store,
    int rank,
    int size,
    const std::chrono::duration<float> &timeout) {
  return c10::make_intrusive<ProcessGroupCNCL>(store, rank, size);
}

}  // namespace torch_mlu
