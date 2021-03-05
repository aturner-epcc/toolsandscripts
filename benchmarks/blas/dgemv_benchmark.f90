program dgemv_benchmark
   use, intrinsic :: iso_fortran_env
   implicit none

   integer, parameter :: dp = REAL64

   integer(4) :: i, j
   integer(8) :: crate, cstart, cend
   real(kind=dp) :: etime, flops
   character(100) :: arg
   
   integer(4) :: n, m, lda, incx, incy
   real(kind=dp) :: alpha, beta
   real(kind=dp), dimension(:,:), allocatable :: a
   real(kind=dp), dimension(:), allocatable :: xn, yn, xt, yt

   ! Read in parameters
   call get_command_argument(1, arg)
   read(arg,*) m
   call get_command_argument(2, arg)
   read(arg,*) n
   call get_command_argument(3, arg)
   read(arg,*) lda
   call get_command_argument(4, arg)
   read(arg,*) incx
   call get_command_argument(5, arg)
   read(arg,*) incy
   call get_command_argument(6, arg)
   read(arg,*) nrun

   ! Allocate and assign arrays
   alpha = 1.0_dp
   beta = 1.0_dp
   allocate(a(lda,n))
   allocate(xn(n))
   allocate(yn(m))
   allocate(xt(m))
   allocate(yt(n))
   do i = 1, n
      do j = 1, m
        a(i,j) = real((i-1)*n + j, dp)
      end do
   end do
   do i = 1, n
      xn(i) = real(i*2 + 1, dp)
      yt(i) = 1.0_dp
   end do
   do j = 1, m
      xt(i) = real(i*2 + 1, dp)
      yn(i) = 1.0_dp
   end do

   ! Warm up
   call dgemv('N', m, n, alpha, a, lda, xn, incx, beta, yn, incy)
   call dgemv('N', m, n, alpha, a, lda, xn, incx, beta, yn, incy)
   call dgemv('N', m, n, alpha, a, lda, xn, incx, beta, yn, incy)
   call dgemv('T', m, n, alpha, a, lda, xt, incx, beta, yt, incy)
   call dgemv('T', m, n, alpha, a, lda, xt, incx, beta, yt, incy)
   call dgemv('T', m, n, alpha, a, lda, xt, incx, beta, yt, incy)
   
   ! Initialise clock
   call system_clock(count_rate=crate)
   write(*,*) "System clock rate = ", crate

   ! Run the benchmark
   call system_clock(cstart)
   do i = 1, nrun
      call dgemv('N', m, n, alpha, a, lda, xn, incx, beta, yn, incy)
   end do
   call system_clock(cend)

   ! Compute execution time
   etime = real(cend - cstart, dp) / real(crate, dp)

   flops = 2.0 * real(n * m, dp)
   flops = real(nrun, dp) * flops / (etime * 1000.0_dp**3)

   write(*,*) "Normal time = ", etime
   write(*,*) "Normal Gflops", flops


   deallocate(a)
   deallocate(xn)
   deallocate(yn)
   deallocate(xt)
   deallocate(yt)

end program dgemv_benchmark