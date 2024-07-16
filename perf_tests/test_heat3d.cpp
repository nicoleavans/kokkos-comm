#include "KokkosComm.hpp"
#include "test_utils.hpp"
#include <iostream>

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace&) {}
  static bool overlap() { return false; }
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(Kokkos::Cuda& space) {
    cudaStream_t stream = space.cuda_stream();
    cudaStreamDestroy(stream);
  }
  static bool overlap() {
    bool value          = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::stoi(local_rank_str) == 0);
    }
    return value;
  }
};
#endif

struct CommHelper {
  MPI_Comm comm;

  int nx, ny, nz; // Num MPI ranks in each dimension
  int me;         // My rank
  int nranks;     // N ranks
  int x, y, z;    // My pos in proc grid

  // Neighbor Ranks
  int up, down, left, right, front, back;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    nx = std::pow(1.0 * nranks, 1.0 / 3.0);
    while (nranks % nx != 0) nx++;
    ny = std::sqrt(1.0 * (nranks / nx));
    while ((nranks / nx) % ny != 0) ny++;

    nz    = nranks / nx / ny;
    x     = me % nx;
    y     = (me / nx) % ny;
    z     = (me / nx / ny);
    left  = x == 0 ? -1 : me - 1;
    right = x == nx - 1 ? -1 : me + 1;
    down  = y == 0 ? -1 : me - nx;
    up    = y == ny - 1 ? -1 : me + nx;
    front = z == 0 ? -1 : me - nx * ny;
    back  = z == nz - 1 ? -1 : me + nx * ny;
  }

  template <class ViewType>
  void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer,
                    MPI_Request* request_send, MPI_Request* request_recv) { 
    MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
    MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
  }

  template <typename ExecSpace, class ViewType>
  void isend_irecv_dc(const ExecSpace &space, const ViewType &sv, ViewType &rv, int src, int dest, int tag, KokkosComm::Req request_send, KokkosComm::Req request_recv){ 
    request_send = KokkosComm::isend<KokkosComm::Impl::Packer::DeepCopy<ViewType>>(space, sv, dest, tag, comm);
    request_recv = KokkosComm::irecv<KokkosComm::Impl::Packer::DeepCopy<ViewType>>(space, rv, src, tag, comm, NULL); // request_send.wait(); request_recv.wait();
  }

  template <typename ExecSpace, class ViewType>
  void isend_irecv_dt(const ExecSpace &space, const ViewType &sv, ViewType &rv, int src, int dest, int tag, KokkosComm::Req request_send, KokkosComm::Req request_recv){ 
    request_send = KokkosComm::isend<KokkosComm::Impl::Packer::MpiDatatype<ViewType>>(space, sv, dest, tag, comm);
    request_recv = KokkosComm::irecv<KokkosComm::Impl::Packer::MpiDatatype<ViewType>>(space, rv, src, tag, comm, NULL); // request_send.wait(); request_recv.wait();
  }
};

struct System {
  // Communicator
  CommHelper comm;
  MPI_Request mpi_requests_recv[6];
  MPI_Request mpi_requests_send[6];
  int mpi_active_requests;

  // size of system
  int X, Y, Z;
  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;

  int N; // number of timesteps
  int I; // interval for print

  // Temperature and delta Temperature
  Kokkos::View<double***> T, dT;
  // Halo data
  using buffer_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  buffer_t T_left, T_right, T_up, T_down, T_front, T_back;
  buffer_t T_left_out, T_right_out, T_up_out, T_down_out, T_front_out, T_back_out;

  Kokkos::DefaultExecutionSpace E_left, E_right, E_up, E_down, E_front, E_back, E_bulk;

  double T0;    // Initial temperature
  double dt;    // timestep width
  double q;     // thermal transfer coefficient
  double sigma; // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double P;     // incoming power

  // init_system
  System(MPI_Comm comm_) : comm(comm_) {
    mpi_active_requests = 0;
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi = Y_hi = Z_hi = X;
    N                  = 5; //10000 reduced for quick testing
    I = N - 1;
    T       = Kokkos::View<double***>();
    dT      = Kokkos::View<double***>();
    T0      = 0.0;
    dt      = 0.1;
    q       = 1.0;
    sigma   = 1.0;
    P       = 1.0;
    auto exec_inst =
      Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1);
    E_left  = exec_inst[left];
    E_right = exec_inst[right];
    E_up    = exec_inst[up];
    E_down  = exec_inst[down];
    E_front = exec_inst[front];
    E_back  = exec_inst[back];
    E_bulk  = exec_inst[bulk];
  }

  void setup_subdomain() {
    int dX = (X + comm.nx - 1) / comm.nx;
    X_lo   = dX * comm.x; X_hi   = X_lo + dX;
    if (X_hi > X) X_hi = X;
    int dY = (Y + comm.ny - 1) / comm.ny;
    Y_lo   = dY * comm.y; Y_hi   = Y_lo + dY;
    if (Y_hi > Y) Y_hi = Y;
    int dZ = (Z + comm.nz - 1) / comm.nz;
    Z_lo   = dZ * comm.z; Z_hi   = Z_lo + dZ;
    if (Z_hi > Z) Z_hi = Z;
    T  = Kokkos::View<double***>("System::T", X_hi - X_lo, Y_hi - Y_lo, Z_hi - Z_lo);
    dT = Kokkos::View<double***>("System::dT", T.extent(0), T.extent(1), T.extent(2));
    Kokkos::deep_copy(T, T0);

    // incoming halos
    if (X_lo != 0) T_left = buffer_t("System::T_left", Y_hi - Y_lo, Z_hi - Z_lo);
    if (X_hi != X) T_right = buffer_t("System::T_right", Y_hi - Y_lo, Z_hi - Z_lo);
    if (Y_lo != 0) T_down = buffer_t("System::T_down", X_hi - X_lo, Z_hi - Z_lo);
    if (Y_hi != Y) T_up = buffer_t("System::T_up", X_hi - X_lo, Z_hi - Z_lo);
    if (Z_lo != 0) T_front = buffer_t("System::T_front", X_hi - X_lo, Y_hi - Y_lo);
    if (Z_hi != Z) T_back = buffer_t("System::T_back", X_hi - X_lo, Y_hi - Y_lo);
    // outgoing halo
    if (X_lo != 0) T_left_out = buffer_t("System::T_left_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if (X_hi != X) T_right_out = buffer_t("System::T_right_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if (Y_lo != 0) T_down_out = buffer_t("System::T_down_out", X_hi - X_lo, Z_hi - Z_lo);
    if (Y_hi != Y) T_up_out = buffer_t("System::T_up_out", X_hi - X_lo, Z_hi - Z_lo);
    if (Z_lo != 0) T_front_out = buffer_t("System::T_front_out", X_hi - X_lo, Y_hi - Y_lo);
    if (Z_hi != Z) T_back_out = buffer_t("System::T_back_out", X_hi - X_lo, Y_hi - Y_lo);
  }

  // run_time_loops
  void timestep() {
    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0;
      pack_T_halo();       // Overlap O1
      compute_inner_dT();  // Overlap O1
      Kokkos::fence();
      exchange_T_halo();
      compute_surface_dT();
      Kokkos::fence();
    }
  }

  // Compute inner update
  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    dT(x, y, z) = dT_xyz;
  }
  void compute_inner_dT() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = T.extent(0); int myY = T.extent(1); int myZ = T.extent(2);
    Kokkos::parallel_for(
        "ComputeInnerDT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
  };

  // Compute non-exposed surface, dispatch makes sure that we don't hit elements twice
  enum { left, right, down, up, front, back, bulk };

  template <int Surface>
  struct ComputeSurfaceDT {};

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i, int j) const {
    int NX = T.extent(0); int NY = T.extent(1); int NZ = T.extent(2);
    int x, y, z;
    if (Surface == left) { x = 0; y = i; z = j; }
    if (Surface == right) { x = NX - 1; y = i; z = j; }
    if (Surface == down) { x = i; y = 0; z = j; }
    if (Surface == up) { x = i; y = NY - 1; z = j; }
    if (Surface == front) { x = i; y = j; z = 0; }
    if (Surface == back) { x = i; y = j; z = NZ - 1; }

    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);

    // Heat conduction to inner body
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < NX - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < NY - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < NZ - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction with Halo
    if (x == 0 && X_lo != 0) dT_xyz += q * (T_left(y, z) - T_xyz);
    if (x == (NX - 1) && X_hi != X) dT_xyz += q * (T_right(y, z) - T_xyz);
    if (y == 0 && Y_lo != 0) dT_xyz += q * (T_down(x, z) - T_xyz);
    if (y == (NY - 1) && Y_hi != Y) dT_xyz += q * (T_up(x, z) - T_xyz);
    if (z == 0 && Z_lo != 0) dT_xyz += q * (T_front(x, y) - T_xyz);
    if (z == (NZ - 1) && Z_hi != Z) dT_xyz += q * (T_back(x, y) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (NX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) +
                       ((y == (NY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) +
                       ((z == (NZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  void pack_T_halo() {
    mpi_active_requests = 0;
    int mar             = 0;
    if (X_lo != 0) {
      Kokkos::deep_copy(E_left, T_left_out, Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL));
      mar++;
    }
    if (Y_lo != 0) {
      Kokkos::deep_copy(E_down, T_down_out, Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL));
      mar++;
    }
    if (Z_lo != 0) {
      Kokkos::deep_copy(E_front, T_front_out, Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0));
      mar++;
    }
    if (X_hi != X) {
      Kokkos::deep_copy(E_right, T_right_out, Kokkos::subview(T, X_hi - X_lo - 1, Kokkos::ALL, Kokkos::ALL));
      mar++;
    }
    if (Y_hi != Y) {
      Kokkos::deep_copy(E_up, T_up_out, Kokkos::subview(T, Kokkos::ALL, Y_hi - Y_lo - 1, Kokkos::ALL));
      mar++;
    }
    if (Z_hi != Z) {
      Kokkos::deep_copy(E_back, T_back_out, Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, Z_hi - Z_lo - 1));
      mar++;
    }
  }

  void exchange_T_halo() {
    int mar = 0;
    if (X_lo != 0) {
      E_left.fence();
      comm.isend_irecv(comm.left, T_left_out, T_left, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      mar++;
    }
    if (Y_lo != 0) {
      E_down.fence();
      comm.isend_irecv(comm.down, T_down_out, T_down, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      mar++;
    }
    if (Z_lo != 0) {
      E_front.fence();
      comm.isend_irecv(comm.front, T_front_out, T_front, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      mar++;
    }
    if (X_hi != X) {
      E_right.fence();
      comm.isend_irecv(comm.right, T_right_out, T_right, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      mar++;
    }
    if (Y_hi != Y) {
      E_up.fence();
      comm.isend_irecv(comm.up, T_up_out, T_up, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      mar++;
    }
    if (Z_hi != Z) {
      E_back.fence();
      comm.isend_irecv(comm.back, T_back_out, T_back, &mpi_requests_send[mar], &mpi_requests_recv[mar]);
      mar++;
    }
    mpi_active_requests = mar;
  }

  void compute_surface_dT() {
    using policy_left_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<left>, int>;
    using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<right>, int>;
    using policy_down_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<down>, int>;
    using policy_up_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<up>, int>;
    using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<front>, int>;
    using policy_back_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<back>, int>;
    int x = T.extent(0); int y = T.extent(1); int z = T.extent(2);
    if (mpi_active_requests > 0) {
      MPI_Waitall(mpi_active_requests, mpi_requests_send, MPI_STATUSES_IGNORE);
      MPI_Waitall(mpi_active_requests, mpi_requests_recv, MPI_STATUSES_IGNORE);
    }
    
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Left",
      Kokkos::Experimental::require(
        policy_left_t(E_left, {0, 0}, {y, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Right",
      Kokkos::Experimental::require(
        policy_right_t(E_right, {0, 0}, {y, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Down",
      Kokkos::Experimental::require(
        policy_down_t(E_down, {1, 0}, {x - 1, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Up",
      Kokkos::Experimental::require(
        policy_up_t(E_up, {1, 0}, {x - 1, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_front",
      Kokkos::Experimental::require(
        policy_front_t(E_front, {1, 1}, {x - 1, y - 1}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_back",
      Kokkos::Experimental::require(
        policy_back_t(E_back, {1, 1}, {x - 1, y - 1}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
  }

  // Some compilers have deduction issues if this were just a tagged operator, so a full Functor here instead
  struct UpdateT {
    Kokkos::View<double***> T, dT;
    double dt;
    UpdateT(Kokkos::View<double***> T_, Kokkos::View<double***> dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_T() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    int x = T.extent(0); int y = T.extent(1); int z = T.extent(2);
    double my_T = 0.0;
    Kokkos::parallel_reduce(
        "UpdateT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {0, 0, 0}, {x, y, z}, {10, 10, 10}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        UpdateT(T, dT, dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return sum_T;
  }
};

struct SystemKC_DC {
  // Communicator
  CommHelper comm;
  KokkosComm::Req kc_requests_recv[6];
  KokkosComm::Req kc_requests_send[6];
  int mpi_active_requests;

  // size of system
  int X, Y, Z;
  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;

  int N; // number of timesteps
  int I; // interval for print

  // Temperature and delta Temperature
  using DataView = Kokkos::View<double *** /*, Kokkos::HostSpace*/>;
  DataView T, dT;

  // Halo data
  using lr_buffer_t = Kokkos::Subview<DataView, int, decltype(Kokkos::ALL), decltype(Kokkos::ALL)>;
  using ud_buffer_t = Kokkos::Subview<DataView, decltype(Kokkos::ALL), int, decltype(Kokkos::ALL)>;
  using fb_buffer_t = Kokkos::Subview<DataView, decltype(Kokkos::ALL), decltype(Kokkos::ALL), int>;
  lr_buffer_t T_left, T_right; ud_buffer_t T_up, T_down; fb_buffer_t T_front, T_back;
  lr_buffer_t T_left_out, T_right_out; ud_buffer_t T_up_out, T_down_out; fb_buffer_t T_front_out, T_back_out;

  Kokkos::DefaultExecutionSpace E_left, E_right, E_up, E_down, E_front, E_back, E_bulk;

  double T0;    // Initial temperature
  double dt;    // timestep width
  double q;     // thermal transfer coefficient
  double sigma; // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double P;     // incoming power

  // init_system
  SystemKC_DC(MPI_Comm comm_) : comm(comm_) {
    mpi_active_requests = 0;
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi = Y_hi = Z_hi = X;
    N                  = 5; //10000 reduced for quick testing
    I = N - 1;
    T       = Kokkos::View<double***>();
    dT      = Kokkos::View<double***>();
    T0      = 0.0;
    dt      = 0.1;
    q       = 1.0;
    sigma   = 1.0;
    P       = 1.0;
    auto exec_inst =
      Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1);
    E_left  = exec_inst[left];
    E_right = exec_inst[right];
    E_up    = exec_inst[up];
    E_down  = exec_inst[down];
    E_front = exec_inst[front];
    E_back  = exec_inst[back];
    E_bulk  = exec_inst[bulk];
  }

  void setup_subdomain() {
    int dX = (X + comm.nx - 1) / comm.nx;
    X_lo   = dX * comm.x; X_hi   = X_lo + dX;
    if (X_hi > X) X_hi = X;
    int dY = (Y + comm.ny - 1) / comm.ny;
    Y_lo   = dY * comm.y; Y_hi   = Y_lo + dY;
    if (Y_hi > Y) Y_hi = Y;
    int dZ = (Z + comm.nz - 1) / comm.nz;
    Z_lo   = dZ * comm.z; Z_hi   = Z_lo + dZ;
    if (Z_hi > Z) Z_hi = Z;
    T  = Kokkos::View<double***>("System::T", X_hi - X_lo, Y_hi - Y_lo, Z_hi - Z_lo);
    dT = Kokkos::View<double***>("System::dT", T.extent(0), T.extent(1), T.extent(2));
    Kokkos::deep_copy(T, T0);

    // incoming halos
    if (X_lo != 0) T_left = Kokkos::subview(T, T.extent(0) - 1, Kokkos::ALL, Kokkos::ALL);
    if (X_hi != X) T_right = Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL);
    if (Y_lo != 0) T_down = Kokkos::subview(T, Kokkos::ALL, T.extent(1) - 1, Kokkos::ALL);
    if (Y_hi != Y) T_up = Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL);
    if (Z_lo != 0) T_front = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, T.extent(2) - 1);
    if (Z_hi != Z) T_back = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0);
    // outgoing halos
    if (X_lo != 0) T_left_out = Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL);
    if (X_hi != X) T_right_out = Kokkos::subview(T, T.extent(0)-1, Kokkos::ALL, Kokkos::ALL);
    if (Y_lo != 0) T_down_out = Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL);
    if (Y_hi != Y) T_up_out = Kokkos::subview(T, Kokkos::ALL, T.extent(1) - 1, Kokkos::ALL);
    if (Z_lo != 0) T_front_out = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0);
    if (Z_hi != Z) T_back_out = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, T.extent(2) - 1);
  }

  // run_time_loops
  void timestep() {
    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0;
      pack_T_halo();       // Overlap O1
      compute_inner_dT();  // Overlap O1
      Kokkos::fence();
      exchange_T_halo();
      compute_surface_dT();
      Kokkos::fence();
    }
  }

  // Compute inner update
  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    dT(x, y, z) = dT_xyz;
  }
  void compute_inner_dT() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = T.extent(0); int myY = T.extent(1); int myZ = T.extent(2);
    Kokkos::parallel_for(
        "ComputeInnerDT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
  };

  // Compute non-exposed surface, dispatch makes sure that we don't hit elements twice
  enum { left, right, down, up, front, back, bulk };

  template <int Surface>
  struct ComputeSurfaceDT {};

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i, int j) const {
    int NX = T.extent(0); int NY = T.extent(1); int NZ = T.extent(2);
    int x, y, z;
    if (Surface == left) { x = 0; y = i; z = j; }
    if (Surface == right) { x = NX - 1; y = i; z = j; }
    if (Surface == down) { x = i; y = 0; z = j; }
    if (Surface == up) { x = i; y = NY - 1; z = j; }
    if (Surface == front) { x = i; y = j; z = 0; }
    if (Surface == back) { x = i; y = j; z = NZ - 1; }

    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);

    // Heat conduction to inner body
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < NX - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < NY - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < NZ - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction with Halo
    if (x == 0 && X_lo != 0) dT_xyz += q * (T_left(y, z) - T_xyz);
    if (x == (NX - 1) && X_hi != X) dT_xyz += q * (T_right(y, z) - T_xyz);
    if (y == 0 && Y_lo != 0) dT_xyz += q * (T_down(x, z) - T_xyz);
    if (y == (NY - 1) && Y_hi != Y) dT_xyz += q * (T_up(x, z) - T_xyz);
    if (z == 0 && Z_lo != 0) dT_xyz += q * (T_front(x, y) - T_xyz);
    if (z == (NZ - 1) && Z_hi != Z) dT_xyz += q * (T_back(x, y) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (NX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) +
                       ((y == (NY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) +
                       ((z == (NZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  void pack_T_halo() {
    mpi_active_requests = 0;
    int mar             = 0;
    KokkosComm::Req request_send, request_recv;
    if (X_lo != 0) {
      comm.isend_irecv_dc(E_left, T_left_out, T_left, comm.left, comm.left, 0, request_send, request_recv);
      mar++;
    }
    if (Y_lo != 0) {
      comm.isend_irecv_dc(E_down, T_down_out, T_down, comm.down, comm.down, 0, request_send, request_recv);
      mar++;
    }
    if (Z_lo != 0) {
      comm.isend_irecv_dc(E_front, T_front_out, T_front, comm.front, comm.front, 0, request_send, request_recv);
      mar++;
    }
    if (X_hi != X) {
      comm.isend_irecv_dc(E_right, T_right_out, T_right, comm.right, comm.right, 0, request_send, request_recv);
      mar++;
    }
    if (Y_hi != Y) {
      comm.isend_irecv_dc(E_up, T_up_out, T_up, comm.up, comm.up, 0, request_send, request_recv);
      mar++;
    }
    if (Z_hi != Z) {
      comm.isend_irecv_dc(E_back, T_back_out, T_back, comm.back, comm.back, 0, request_send, request_recv);
      mar++;
    }
  }

  void exchange_T_halo() {
    int mar = 0;
    KokkosComm::Req request_send, request_recv;
    if (X_lo != 0) {
      E_left.fence();
      comm.isend_irecv_dc(E_left, T_left_out, T_left, comm.left, comm.left, 0, request_send, request_recv);
      mar++;
    }
    if (Y_lo != 0) {
      E_down.fence();
      comm.isend_irecv_dc(E_down, T_down_out, T_down, comm.down, comm.down, 0, request_send, request_recv);
      mar++;
    }
    if (Z_lo != 0) {
      E_front.fence();
      comm.isend_irecv_dc(E_front, T_front_out, T_front, comm.front, comm.front, 0, request_send, request_recv);
      mar++;
    }
    if (X_hi != X) {
      E_right.fence();
      comm.isend_irecv_dc(E_right, T_right_out, T_right, comm.right, comm.right, 0, request_send, request_recv);
      mar++;
    }
    if (Y_hi != Y) {
      E_up.fence();
      comm.isend_irecv_dc(E_up, T_up_out, T_up, comm.up, comm.up, 0, request_send, request_recv);
      mar++;
    }
    if (Z_hi != Z) {
      E_back.fence();
      comm.isend_irecv_dc(E_back, T_back_out, T_back, comm.back, comm.back, 0, request_send, request_recv);
      mar++;
    }
    mpi_active_requests = mar;
  }

  void compute_surface_dT() {
    using policy_left_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<left>, int>;
    using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<right>, int>;
    using policy_down_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<down>, int>;
    using policy_up_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<up>, int>;
    using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<front>, int>;
    using policy_back_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<back>, int>;
    int x = T.extent(0); int y = T.extent(1); int z = T.extent(2);
    if (mpi_active_requests > 0) {
      for(int i = 0; i < mpi_active_requests; i++){

        kc_requests_recv[i].wait();
        kc_requests_send[i].wait();
      }
    }
    
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Left",
      Kokkos::Experimental::require(
        policy_left_t(E_left, {0, 0}, {y, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Right",
      Kokkos::Experimental::require(
        policy_right_t(E_right, {0, 0}, {y, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Down",
      Kokkos::Experimental::require(
        policy_down_t(E_down, {1, 0}, {x - 1, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Up",
      Kokkos::Experimental::require(
        policy_up_t(E_up, {1, 0}, {x - 1, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_front",
      Kokkos::Experimental::require(
        policy_front_t(E_front, {1, 1}, {x - 1, y - 1}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_back",
      Kokkos::Experimental::require(
        policy_back_t(E_back, {1, 1}, {x - 1, y - 1}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
  }

  // Some compilers have deduction issues if this were just a tagged operator, so a full Functor here instead
  struct UpdateT {
    Kokkos::View<double***> T, dT;
    double dt;
    UpdateT(Kokkos::View<double***> T_, Kokkos::View<double***> dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_T() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    int x = T.extent(0); int y = T.extent(1); int z = T.extent(2);
    double my_T = 0.0;
    Kokkos::parallel_reduce(
        "UpdateT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {0, 0, 0}, {x, y, z}, {10, 10, 10}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        UpdateT(T, dT, dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return sum_T;
  }
};

struct SystemKC_MPIDT {
  // Communicator
  CommHelper comm;
  KokkosComm::Req kc_requests_recv[6];
  KokkosComm::Req kc_requests_send[6];
  int mpi_active_requests;

  // size of system
  int X, Y, Z;
  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;

  int N; // number of timesteps
  int I; // interval for print

  // Temperature and delta Temperature
  using DataView = Kokkos::View<double *** /*, Kokkos::HostSpace*/>;
  DataView T, dT;

  // Halo data
  using lr_buffer_t = Kokkos::Subview<DataView, int, decltype(Kokkos::ALL), decltype(Kokkos::ALL)>;
  using ud_buffer_t = Kokkos::Subview<DataView, decltype(Kokkos::ALL), int, decltype(Kokkos::ALL)>;
  using fb_buffer_t = Kokkos::Subview<DataView, decltype(Kokkos::ALL), decltype(Kokkos::ALL), int>;
  lr_buffer_t T_left, T_right;
  ud_buffer_t T_up, T_down;
  fb_buffer_t T_front, T_back;

  lr_buffer_t T_left_out, T_right_out;
  ud_buffer_t T_up_out, T_down_out;
  fb_buffer_t T_front_out, T_back_out;

  Kokkos::DefaultExecutionSpace E_left, E_right, E_up, E_down, E_front, E_back, E_bulk;

  double T0;    // Initial temperature
  double dt;    // timestep width
  double q;     // thermal transfer coefficient
  double sigma; // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double P;     // incoming power

  // init_system
  SystemKC_MPIDT(MPI_Comm comm_) : comm(comm_) {
    mpi_active_requests = 0;
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi = Y_hi = Z_hi = X;
    N                  = 5; //10000 reduced for quick testing
    I = N - 1;
    T       = Kokkos::View<double***>();
    dT      = Kokkos::View<double***>();
    T0      = 0.0;
    dt      = 0.1;
    q       = 1.0;
    sigma   = 1.0;
    P       = 1.0;
    auto exec_inst =
      Kokkos::Experimental::partition_space(Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1, 1, 1);
    E_left  = exec_inst[left];
    E_right = exec_inst[right];
    E_up    = exec_inst[up];
    E_down  = exec_inst[down];
    E_front = exec_inst[front];
    E_back  = exec_inst[back];
    E_bulk  = exec_inst[bulk];
  }

  void setup_subdomain() {
    int dX = (X + comm.nx - 1) / comm.nx;
    X_lo   = dX * comm.x; X_hi   = X_lo + dX;
    if (X_hi > X) X_hi = X;
    int dY = (Y + comm.ny - 1) / comm.ny;
    Y_lo   = dY * comm.y; Y_hi   = Y_lo + dY;
    if (Y_hi > Y) Y_hi = Y;
    int dZ = (Z + comm.nz - 1) / comm.nz;
    Z_lo   = dZ * comm.z; Z_hi   = Z_lo + dZ;
    if (Z_hi > Z) Z_hi = Z;
    T  = Kokkos::View<double***>("System::T", X_hi - X_lo, Y_hi - Y_lo, Z_hi - Z_lo);
    dT = Kokkos::View<double***>("System::dT", T.extent(0), T.extent(1), T.extent(2));
    Kokkos::deep_copy(T, T0);

    // incoming halos
    if (X_lo != 0) T_left = Kokkos::subview(T, T.extent(0) - 1, Kokkos::ALL, Kokkos::ALL);
    if (X_hi != X) T_right = Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL);
    if (Y_lo != 0) T_down = Kokkos::subview(T, Kokkos::ALL, T.extent(1) - 1, Kokkos::ALL);
    if (Y_hi != Y) T_up = Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL);
    if (Z_lo != 0) T_front = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, T.extent(2) - 1);
    if (Z_hi != Z) T_back = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0);
    // outgoing halos
    if (X_lo != 0) T_left_out = Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL);
    if (X_hi != X) T_right_out = Kokkos::subview(T, T.extent(0)-1, Kokkos::ALL, Kokkos::ALL);
    if (Y_lo != 0) T_down_out = Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL);
    if (Y_hi != Y) T_up_out = Kokkos::subview(T, Kokkos::ALL, T.extent(1) - 1, Kokkos::ALL);
    if (Z_lo != 0) T_front_out = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0);
    if (Z_hi != Z) T_back_out = Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, T.extent(2) - 1);
  }

  // run_time_loops
  void timestep() {
    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0;
      pack_T_halo();       // Overlap O1
      compute_inner_dT();  // Overlap O1
      Kokkos::fence();
      exchange_T_halo();
      compute_surface_dT();
      Kokkos::fence();
    }
  }

  // Compute inner update
  struct ComputeInnerDT {};

  KOKKOS_FUNCTION
  void operator()(ComputeInnerDT, int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    dT(x, y, z) = dT_xyz;
  }
  void compute_inner_dT() {
    using policy_t =
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, ComputeInnerDT, int>;
    int myX = T.extent(0); int myY = T.extent(1); int myZ = T.extent(2);
    Kokkos::parallel_for(
        "ComputeInnerDT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
  };

  // Compute non-exposed surface, dispatch makes sure that we don't hit elements twice
  enum { left, right, down, up, front, back, bulk };

  template <int Surface>
  struct ComputeSurfaceDT {};

  template <int Surface>
  KOKKOS_FUNCTION void operator()(ComputeSurfaceDT<Surface>, int i, int j) const {
    int NX = T.extent(0); int NY = T.extent(1); int NZ = T.extent(2);
    int x, y, z;
    if (Surface == left) { x = 0; y = i; z = j; }
    if (Surface == right) { x = NX - 1; y = i; z = j; }
    if (Surface == down) { x = i; y = 0; z = j; }
    if (Surface == up) { x = i; y = NY - 1; z = j; }
    if (Surface == front) { x = i; y = j; z = 0; }
    if (Surface == back) { x = i; y = j; z = NZ - 1; }

    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);

    // Heat conduction to inner body
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < NX - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < NY - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < NZ - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction with Halo
    if (x == 0 && X_lo != 0) dT_xyz += q * (T_left(y, z) - T_xyz);
    if (x == (NX - 1) && X_hi != X) dT_xyz += q * (T_right(y, z) - T_xyz);
    if (y == 0 && Y_lo != 0) dT_xyz += q * (T_down(x, z) - T_xyz);
    if (y == (NY - 1) && Y_hi != Y) dT_xyz += q * (T_up(x, z) - T_xyz);
    if (z == 0 && Z_lo != 0) dT_xyz += q * (T_front(x, y) - T_xyz);
    if (z == (NZ - 1) && Z_hi != Z) dT_xyz += q * (T_back(x, y) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) +
                       ((x == (NX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) +
                       ((y == (NY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) +
                       ((z == (NZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }

  void pack_T_halo() {
    mpi_active_requests = 0;
    int mar             = 0;
    KokkosComm::Req request_send, request_recv;
    if (X_lo != 0) {
      comm.isend_irecv_dt(E_left, T_left_out, T_left, comm.left, comm.left, 0, request_send, request_recv);
      mar++;
    }
    if (Y_lo != 0) {
      comm.isend_irecv_dt(E_down, T_down_out, T_down, comm.down, comm.down, 0, request_send, request_recv);
      mar++;
    }
    if (Z_lo != 0) {
      comm.isend_irecv_dt(E_front, T_front_out, T_front, comm.front, comm.front, 0, request_send, request_recv);
      mar++;
    }
    if (X_hi != X) {
      comm.isend_irecv_dt(E_right, T_right_out, T_right, comm.right, comm.right, 0, request_send, request_recv);
      mar++;
    }
    if (Y_hi != Y) {
      comm.isend_irecv_dt(E_up, T_up_out, T_up, comm.up, comm.up, 0, request_send, request_recv);
      mar++;
    }
    if (Z_hi != Z) {
      comm.isend_irecv_dt(E_back, T_back_out, T_back, comm.back, comm.back, 0, request_send, request_recv);
      mar++;
    }
  }

  void exchange_T_halo() {
    int mar = 0;
    KokkosComm::Req request_send, request_recv;
    if (X_lo != 0) {
      E_left.fence();
      comm.isend_irecv_dt(E_left, T_left_out, T_left, comm.left, comm.left, 0, request_send, request_recv);
      mar++;
    }
    if (Y_lo != 0) {
      E_down.fence();
      comm.isend_irecv_dt(E_down, T_down_out, T_down, comm.down, comm.down, 0, request_send, request_recv);
      mar++;
    }
    if (Z_lo != 0) {
      E_front.fence();
      comm.isend_irecv_dt(E_front, T_front_out, T_front, comm.front, comm.front, 0, request_send, request_recv);
      mar++;
    }
    if (X_hi != X) {
      E_right.fence();
      comm.isend_irecv_dt(E_right, T_right_out, T_right, comm.right, comm.right, 0, request_send, request_recv);
      mar++;
    }
    if (Y_hi != Y) {
      E_up.fence();
      comm.isend_irecv_dt(E_up, T_up_out, T_up, comm.up, comm.up, 0, request_send, request_recv);
      mar++;
    }
    if (Z_hi != Z) {
      E_back.fence();
      comm.isend_irecv_dt(E_back, T_back_out, T_back, comm.back, comm.back, 0, request_send, request_recv);
      mar++;
    }
    mpi_active_requests = mar;
  }

  void compute_surface_dT() {
    using policy_left_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<left>, int>;
    using policy_right_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<right>, int>;
    using policy_down_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<down>, int>;
    using policy_up_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<up>, int>;
    using policy_front_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<front>, int>;
    using policy_back_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, ComputeSurfaceDT<back>, int>;
    int x = T.extent(0); int y = T.extent(1); int z = T.extent(2);
    if (mpi_active_requests > 0) {
      for(int i = 0; i < mpi_active_requests; i++){
        kc_requests_recv[i].wait();
        kc_requests_send[i].wait();
      }
    }
    
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Left",
      Kokkos::Experimental::require(
        policy_left_t(E_left, {0, 0}, {y, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Right",
      Kokkos::Experimental::require(
        policy_right_t(E_right, {0, 0}, {y, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Down",
      Kokkos::Experimental::require(
        policy_down_t(E_down, {1, 0}, {x - 1, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_Up",
      Kokkos::Experimental::require(
        policy_up_t(E_up, {1, 0}, {x - 1, z}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_front",
      Kokkos::Experimental::require(
        policy_front_t(E_front, {1, 1}, {x - 1, y - 1}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
    Kokkos::parallel_for(
      "ComputeSurfaceDT_back",
      Kokkos::Experimental::require(
        policy_back_t(E_back, {1, 1}, {x - 1, y - 1}),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight), *this);
  }

  // Some compilers have deduction issues if this were just a tagged operator, so a full Functor here instead
  struct UpdateT {
    Kokkos::View<double***> T, dT;
    double dt;
    UpdateT(Kokkos::View<double***> T_, Kokkos::View<double***> dT_, double dt_)
        : T(T_), dT(dT_), dt(dt_) {}
    KOKKOS_FUNCTION
    void operator()(int x, int y, int z, double& sum_T) const {
      sum_T += T(x, y, z);
      T(x, y, z) += dt * dT(x, y, z);
    }
  };

  double update_T() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    int x = T.extent(0); int y = T.extent(1); int z = T.extent(2);
    double my_T = 0.0;
    Kokkos::parallel_reduce(
        "UpdateT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {0, 0, 0}, {x, y, z}, {10, 10, 10}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight),
        UpdateT(T, dT, dt), my_T);
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return sum_T;
  }
};

void benchmark_heat3d_mpi(benchmark::State &state) {
  System sys(MPI_COMM_WORLD);
  sys.setup_subdomain();
  auto f = std::bind(&System::timestep, &sys);
  while(state.KeepRunning()){
    do_iteration(state, MPI_COMM_WORLD, f);
  }
}

void benchmark_heat3d_kc_dc(benchmark::State &state) {
  SystemKC_DC sys(MPI_COMM_WORLD);
  sys.setup_subdomain();
  auto f = std::bind(&SystemKC_DC::timestep, &sys);
  while(state.KeepRunning()){
    do_iteration(state, MPI_COMM_WORLD, f);
  }
}

void benchmark_heat3d_kc_mpidt(benchmark::State &state) {
  SystemKC_MPIDT sys(MPI_COMM_WORLD);
  sys.setup_subdomain();
  auto f = std::bind(&SystemKC_MPIDT::timestep, &sys);
  while(state.KeepRunning()){
    do_iteration(state, MPI_COMM_WORLD, f);
  }
}

BENCHMARK(benchmark_heat3d_mpi)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_heat3d_kc_dc)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_heat3d_kc_mpidt)->UseManualTime()->Unit(benchmark::kMicrosecond);