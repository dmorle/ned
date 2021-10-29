#ifndef NEDVM_COMMON_H
#define NEDVM_COMMON_H

#ifdef _DEBUG
#define DEBUG_WAS_DEFINED
#undef _DEBUG
#endif

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ADT/ArrayRef.h>

#ifdef DEBUG_WAS_DEFINED
#define _DEBUG
#endif

#endif
