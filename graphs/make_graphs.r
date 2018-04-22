library(ggplot2)
alldata = read.csv('results.csv')

grapher <- function(size, ty, wl, title) {
	ftab = subset(subset(subset(alldata, number.of.elements == size), data.type == ty), workload == wl)
	png(filename=paste('graphs/', paste(ty,wl,size,sep='_'), '.png', sep=''), width=600, height=600)
	barplot(ftab$mean.time.per.operation.ns,
		main=title,
		names.arg=ftab$data.structure,
		ylab='ns per operation',
		col=rainbow(20)[c(1,3,5,7)],
		xlab='Data Structure',
		ylim=c(0, 1.1*max(ftab$mean.time.per.operation.ns, na.rm=TRUE)))
	dev.off()
}

bothgraph <- function(size, ty1, ty2, wl, title) {
	ftab1 = subset(subset(subset(alldata, number.of.elements == size), data.type == ty1), workload == wl)
	ftab2 = subset(subset(subset(alldata, number.of.elements == size), data.type == ty2), workload == wl)
	png(filename=paste('graphs/', paste(ty1,ty2,wl,size,sep='_'), '.png', sep=''), width=600, height=600)
	tags = c(rep('dense', length(ftab1$mean.time.per.operation.ns)),
		 rep('sparse', length(ftab1$mean.time.per.operation.ns)))
	btab = data.frame(Data.Structure=rep(ftab1$data.structure, 2),
			  ns.Per.Operation=c(ftab1$mean.time.per.operation.ns, ftab2$mean.time.per.operation.ns),
			  workload=tags)
	ggplot(btab, aes(fill=workload, y=ns.Per.Operation, x=Data.Structure)) +
		geom_bar(position="dodge", stat="identity") +
		labs(title=title, x="Data Structure", y="ns Per Operation")

	# dev.off()
	# print(btab)
	# barplot(btab,
	# 	main=title,
	# 	names.arg=ftab1$data.structure,
	# 	ylab='ns per operation',
	# 	col=rainbow(20)[c(1,7)],
	# 	xlab='Data Structure',
	# 	legend=c('dense', 'sparse'),
	# 	ylim=c(0, 1.1*max(ftab2$mean.time.per.operation.ns, na.rm=TRUE)),
	# 	beside=TRUE)
	# dev.off()
}

bothgraph(16384,     'dense_u64', 'sparse_u64', 'lookup_hit',
	  'Lookups for elements in the set with integer keys, 16K elements')
bothgraph(16777216,  'dense_u64', 'sparse_u64', 'lookup_hit',
	  'Lookups for elements in the set with integer keys, 16M elements')
bothgraph(268435456, 'dense_u64', 'sparse_u64', 'lookup_hit',
	  'Lookups for elements in the set with integer keys, 256M elements')

bothgraph(16384,     'dense_u64', 'sparse_u64', 'lookup_miss',
	  'Lookups for elements not in the set with integer keys, 16K elements')
bothgraph(16777216,  'dense_u64', 'sparse_u64', 'lookup_miss',
	  'Lookups for elements not in the set with integer keys, 16M elements')
bothgraph(268435456, 'dense_u64', 'sparse_u64', 'lookup_miss',
	  'Lookups for elements not in the set with integer keys, 256M elements')

bothgraph(16384,     'dense_u64', 'sparse_u64', 'insert_remove',
	  'Insert/Remove pairs with integer keys, 16K elements')
bothgraph(16777216,  'dense_u64', 'sparse_u64', 'insert_remove',
	  'Insert/Remove pairs with integer keys, 16M elements')
bothgraph(268435456, 'dense_u64', 'sparse_u64', 'insert_remove',
	  'Insert/Remove pairs with integer keys, 256M elements')

grapher(16384, 	   'String', 'lookup_hit', 'Lookups in the set, UTF-8 Strings of mean length 10, 16K elements')
grapher(1048576,   'String', 'lookup_hit', 'Lookups in the set, UTF-8 Strings of mean length 10, 1M elements')
grapher(16777216,  'String', 'lookup_hit', 'Lookups in the set, UTF-8 Strings of mean length 10, 16M elements')

grapher(16384, 	   'String', 'lookup_miss', 'Lookups not in the set, UTF-8 Strings of mean length 10, 16K elements')
grapher(1048576,   'String', 'lookup_miss', 'Lookups not in the set, UTF-8 Strings of mean length 10, 1M elements')
grapher(16777216,  'String', 'lookup_miss', 'Lookups not in the set, UTF-8 Strings of mean length 10, 16M elements')

grapher(16384,     'String', 'insert_remove', 'insert/remove pairs, UTF-8 Strings of mean length 10, 16K elements')
grapher(1048576,   'String', 'insert_remove', 'insert/remove pairs, UTF-8 Strings of mean length 10, 1M elements')
grapher(16777216,  'String', 'insert_remove', 'insert/remove pairs, UTF-8 Strings of mean length 10, 16M elements')
