/*-256 tests: %s\n", sha256_test() ? "SUCCEEDED" : "FAILED");

 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


// TCP
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define BUFSIZE 1024

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif


typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct {
	unsigned char data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX;

#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/*********************************************************************
* Filename:   sha256.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Implementation of the SHA-256 hashing algorithm.
              SHA-256 is one of the three algorithms in the SHA2
              specification. The others, SHA-384 and SHA-512, are not
              offered in this implementation.
              Algorithm specification can be found here:
               * http://csrc.nist.gov/publications/fips/fips180-2/fips180-2withchangenotice.pdf
              This implementation uses little endian byte order.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <memory.h>

/****************************** MACROS ******************************/
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
__device__ static const WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static const WORD h_k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

/*********************** FUNCTION DEFINITIONS ***********************/
 __device__ void sha256_transform(SHA256_CTX *ctx, const unsigned char data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];
	for (i = 0; i < 64; ++i) {
                t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

 __host__ void h_sha256_transform(SHA256_CTX *ctx, const unsigned char data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];
	for (i = 0; i < 64; ++i) {
                t1 = h + EP1(e) + CH(e,f,g) + h_k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}
	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}
__device__ void sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void sha256_update(SHA256_CTX *ctx, const char data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void sha256_final(SHA256_CTX *ctx, char hash[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}


__host__ void h_sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__host__ void h_sha256_update(SHA256_CTX *ctx, const char data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			h_sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__host__ void h_sha256_final(SHA256_CTX *ctx, char hash[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		h_sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	h_sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}


__host__ __device__ void print_hash(unsigned char * buf)
{
   for (int i = 0; i < 32; i++){
      unsigned int hexnum = (unsigned int) buf[i];
      printf("%02x", hexnum);
   } 
   printf("\n");
   return;
}
__host__ __device__ int strLength(char * str)
{

   int count = 0;
   while (str[count] != '\n')
       count ++;
   printf("\nStrlen of %d", count);
   return count;
}

__host__ __device__ void strcpy(char * a, char * b)
{
   for (int i = 0; i < strLength(b); i++){
       a[i] = b[i];
       a[i + 1] = '\0';
   }


}
__device__ void sha256_hash(unsigned char * str)
{
	unsigned char text1[] = {"000000004c6fe27a1151135df1b1f5d36bc37b6455106e2fc64a8affb4518ddc"}; 
//	char text2[] = {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
//	char text3[] = {"aaaaaaaaaa"};
//	char hash1[SHA256_BLOCK_SIZE] = {0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
//	                                 0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
//	char hash2[SHA256_BLOCK_SIZE] = {0x24,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,
//	                                 0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1};
//	char hash3[SHA256_BLOCK_SIZE] = {0xcd,0xc7,0x6e,0x5c,0x99,0x14,0xfb,0x92,0x81,0xa1,0xc7,0xe2,0x84,0xd7,0x3e,0x67,
//	                                 0xf1,0x80,0x9a,0x48,0xa4,0x97,0x20,0x0e,0x04,0x6d,0x39,0xcc,0xc7,0x11,0x2c,0xd0};
      	unsigned char buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;
//        str  = "00000000308971eee4b34bf76a3eda47bbfbdf1d0cf407a5ed6daf182f4b23b8"; 
	//int idx;
	//int pass = 1;
        
        printf("Starting sha hash with string -%s-\n", str);
        print_hash(str);
        printf("HERE");
        char hash_str[100];
        int n = 0;
        while (str[n] != '\0' || n < SHA256_BLOCK_SIZE)
        {
            hash_str[n] = (char) str[n];
            n ++;
        }
        str[n] = '\0';
        printf("Copied %d bytes\n", n);
        printf("Here is the hash_str\n");
        for (int i = 0; i < n; i ++)
            printf("%d-", hash_str[i]);
//        printf(hash_str);
//        printf("\n");
        //print_hash((unsigned char *)hash_str);
	//print_hash(str);
//        printf("\nHash length - %d", strLength((char*)hash_str));
        sha256_init(&ctx);
        printf("Finishing init");
	sha256_update(&ctx, (char*)text1, SHA256_BLOCK_SIZE );
        printf("Finishing update");
	sha256_final(&ctx, (char *) buf);
        printf("Finished hash");
        int difficulty = 33;
        bool invalid = false;
        //printf("%x", buf);
	for (int i = 0; i < 32; i ++){
//           printf("%c", (unsigned char*) buf[i]);
           unsigned int hexnum = (unsigned int) buf[i];
//           printf("%x-", (unsigned int) hexnum );
           for (int j = 128; j >= 1; j= j / 2){
//              printf("%d", hexnum & j);
              if (((int)hexnum & j) != 0){
           //      printf("1");
                 invalid = true;
              } else {
           //      printf("0");
                 difficulty --;
                 if (difficulty == 0)
                      break;
              }
              if (invalid || difficulty == 0)
                  break;
              //printf("%d", ((unsigned int) buf[i]) & j); 
           }
           if (invalid || difficulty == 0)
                break;
           //printf("%d", (unsigned char*) buf[i]);
           //printf("\nNext Bits\n");
        }
        printf("Printing hash \n");
        print_hash(buf);
        printf("Finished printing hash");
        if (invalid){
             printf("Not enough work done %d\n", difficulty);
             buf[0] = '\0';
            
             memcpy(str, buf, SHA256_BLOCK_SIZE);
        }else{
             printf("YAY you found one");
             print_hash(buf);
             memcpy(str, buf, SHA256_BLOCK_SIZE);
        }
}


__host__ void h_sha256_hash(char * str)
{
      	unsigned char buf[SHA256_BLOCK_SIZE];
	SHA256_CTX ctx;
        printf("Starting sha hash \n");
	h_sha256_init(&ctx);
        printf("Finishing init");
	h_sha256_update(&ctx, str, strLength(str));
        printf("Finishing update");
	h_sha256_final(&ctx, (char *) buf);
        printf("Finished hash");
        int difficulty = 33;
        bool invalid = false;
	for (int i = 0; i < 32; i ++){
           unsigned int hexnum = (unsigned int) buf[i];
           for (int j = 128; j >= 1; j= j / 2){
              if (((int)hexnum & j) != 0){
                 invalid = true;
              } else {
                 difficulty --;
                 if (difficulty == 0)
                      break;
              }
              if (invalid || difficulty == 0)
                  break;
              //printf("%d", ((unsigned int) buf[i]) & j); 
           }
           if (invalid || difficulty == 0)
                break;
           //printf("%d", (unsigned char*) buf[i]);
           //printf("\nNext Bits\n");
        }
        printf("Printing hash \n");
        print_hash(buf);
        printf("Finished printing hash");
        if (invalid){
             printf("Not enough work done %d\n", difficulty);
             buf[0] = '\0';
            
             memcpy(str, buf, SHA256_BLOCK_SIZE);
        }else{
             printf("YAY you found one");
             print_hash(buf);
             memcpy(str, buf, SHA256_BLOCK_SIZE);
        }
}


__global__ void testKernel(unsigned char *var)
{   
    printf("Yay in gpu mode\n");
    print_hash((unsigned char*)var);
//    printf("[%d, %d]:\t\tValue is:%s\n",\
//            blockIdx.y*gridDim.x+blockIdx.x,\
//            threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
//            (char*)var);
    sha256_hash(var);
    printf("End kernel");
}

/* 
 * error - wrapper for perror
 */
void error(char *msg) {
    perror(msg);
    exit(0);
}

void getTip(char * buf){
    int sockfd, portno, n;
    struct sockaddr_in serveraddr;
    struct hostent *server;
    char *hostname;
    //char buf[BUFSIZE];

    /* check command line arguments */
//    if (argc != 3) {
//       fprintf(stderr,"usage: %s <hostname> <port>\n", argv[0]);
      // exit(0);
//    }
//    hostname = argv[1];
    hostname = (char*) "localhost\0";
    hostname = (char*) "hubris.media.mit.edu\0";
//    portno = atoi(argv[2]);
    portno = 6262;

    /* socket: create the socket */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0)
        error((char *) "ERROR opening socket");

    /* gethostbyname: get the server's DNS entry */
    server = gethostbyname(hostname);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host as %s\n", hostname);
        exit(0);
    }

    /* build the server's Internet address */
    bzero((char *) &serveraddr, sizeof(serveraddr));
    serveraddr.sin_family = AF_INET;
    bcopy((char *)server->h_addr,
          (char *)&serveraddr.sin_addr.s_addr, server->h_length);
    serveraddr.sin_port = htons(portno);

    /* connect: create a connection with the server */
    if (connect(sockfd, (struct sockaddr *) &serveraddr, sizeof(serveraddr)) < 0)
      error((char *) "ERROR connecting");

    /* get message line from the user */
//    printf("Please enter msg: ");
    bzero(buf, BUFSIZE);
    //fgets(buf, BUFSIZE, stdin);
    sprintf(buf, "TRQ\n");
    /* send the message line to the server */
    n = write(sockfd, buf, strlen(buf));
    if (n < 0)
      error((char *) "ERROR writing to socket");

    /* print the server's reply */
    bzero(buf, BUFSIZE);
    n = read(sockfd, buf, BUFSIZE);
    if (n < 0)
      error((char *) "ERROR reading from socket");
    printf("Returned tip: %s-----------", buf);
    close(sockfd);
    return;

}

int main(int argc, char **argv)
{
    int devID;
    cudaDeviceProp props;

    // This will pick the best possible CUDA capable device
    devID = findCudaDevice(argc, (const char **)argv);

    //Get GPU information
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    printf("printf() is called. Output:\n\n");

//    printf("SHA-256 tests: %s\n", sha256_test() ? "SUCCEEDED" : "FAILED");


    //Kernel configuration, where a two-dimensional grid and
    //three-dimensional blocks are configured.
//    dim3 dimGrid(2, 2);
//    dim3 dimBlock(2, 2, 2);
//    testKernel<<<dimGrid, dimBlock>>>(10);
    char *tip = (char*) malloc(BUFSIZE);
    unsigned char * h_tip = (unsigned char*) malloc(SHA256_BLOCK_SIZE);
    printf("Gettin tip");
    getTip(tip);
    for (int i=0; i < 100; i++)
       printf("%d_", tip[i]);
    printf("\n");

    h_sha256_hash(tip);
    for (int i= 0; i < SHA256_BLOCK_SIZE; i++)
        printf("%d_", tip[i]);
    printf("\n");
//    h_tip = (unsigned char *) tip;
    memcpy(h_tip, tip, SHA256_BLOCK_SIZE);
    print_hash((unsigned char*)tip);
    print_hash(h_tip);
    unsigned char* d_tip = NULL;
    cudaError_t err = cudaSuccess;    
    err = cudaMalloc((void **)&d_tip, SHA256_BLOCK_SIZE);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    err = cudaMemcpy(d_tip, h_tip, SHA256_BLOCK_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    testKernel<<<1,1>>>(d_tip);
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}

