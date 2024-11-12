#!/bin/sh

src_path=$1
install_path=$2

version=$(grep "^Version:" "${src_path}/.jenkins/package/debian-12/control" | cut -d' ' -f2)
architecture=$(grep "^Architecture:" "${src_path}/.jenkins/package/debian-12/control" | cut -d' ' -f2)

# Create folders.
package_name=marco-${version}_${architecture}

mkdir -p "${package_name}"/DEBIAN
mkdir -p "${package_name}"/usr/bin
mkdir -p "${package_name}"/usr/lib/marco

# Copy the control file.
cp "${src_path}/.jenkins/package/debian-12/control" "${package_name}"/DEBIAN/control

# Copy the driver.
cp "${install_path}/bin/marco" "${package_name}"/usr/lib/marco/marco

# Copy the driver wrapper.
cp "${src_path}/.jenkins/package/debian-12/marco-wrapper.sh" "${package_name}"/usr/bin/marco
chmod +x "${package_name}"/usr/bin/marco

# Build the package.
dpkg-deb --build "${package_name}"

# Clean the work directory.
rm -rf "${package_name}"
